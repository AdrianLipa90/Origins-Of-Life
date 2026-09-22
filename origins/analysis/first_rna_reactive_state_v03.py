"""Explicit reactive-state nucleotide chemistry for first-RNA calibration v0.3.

This module is deliberately coarse grained. It distinguishes free non-reactive
nucleotide equivalents (U), free reactive/activated nucleotide equivalents (A),
and chain populations C_L. It contains no geometry, zeta, sequence-function, or
replication operator.

v0.3 is a calibration architecture test, not a predictive kinetic model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReactiveProfile:
    name: str
    hours: float
    dt_h: float = 0.1
    initial_units: float = 10_000.0
    initial_reactive_fraction: float = 1.0
    max_len: int = 80
    k_activation: float = 0.0
    k_deactivation: float = 1e-4
    k_nucleation: float = 1e-10
    k_extension: float = 9e-7
    k_hydrolysis: float = 1e-5
    cycle_h: float | None = 24.0
    dry_fraction: float = 0.5
    dry_extension_multiplier: float = 10.0
    dry_hydrolysis_multiplier: float = 0.1


CYCLIC_WET_DRY_PROFILE = ReactiveProfile(
    name="cyclic_wet_dry_calibration",
    hours=240.0,
)

ACTIVATED_CLAY_PROFILE = ReactiveProfile(
    name="activated_clay_calibration",
    hours=24.0,
    k_extension=1e-4,
    cycle_h=None,
    dry_fraction=0.0,
    dry_extension_multiplier=1.0,
    dry_hydrolysis_multiplier=1.0,
)


@dataclass
class ReactiveState:
    unreactive_free: float
    reactive_free: float
    chains: np.ndarray
    max_relative_conservation_error: float = 0.0

    @classmethod
    def initialize(cls, profile: ReactiveProfile) -> "ReactiveState":
        if not (0.0 <= profile.initial_reactive_fraction <= 1.0):
            raise ValueError("initial_reactive_fraction must lie in [0,1]")
        chains = np.zeros(profile.max_len + 1, dtype=np.float64)
        reactive = profile.initial_units * profile.initial_reactive_fraction
        unreactive = profile.initial_units - reactive
        return cls(
            unreactive_free=float(unreactive),
            reactive_free=float(reactive),
            chains=chains,
        )

    def total_units(self) -> float:
        lengths = np.arange(len(self.chains), dtype=np.float64)
        return float(
            self.unreactive_free
            + self.reactive_free
            + np.dot(lengths, self.chains)
        )

    def mass_fraction_ge(self, length: int) -> float:
        total = self.total_units()
        if total <= 0:
            return 0.0
        lengths = np.arange(len(self.chains), dtype=np.float64)
        mask = lengths >= int(length)
        return float(np.dot(lengths[mask], self.chains[mask]) / total)

    def max_length_at_mass_fraction(self, threshold: float) -> int:
        total = self.total_units()
        if total <= 0:
            return 0
        for length in range(len(self.chains) - 1, 1, -1):
            frac = float(length * self.chains[length] / total)
            if frac >= threshold:
                return length
        return 0


def _finite_nonnegative(value: float, *, name: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise FloatingPointError(f"{name} became non-finite")
    if x < -1e-12:
        raise FloatingPointError(f"{name} became negative: {x}")
    return max(0.0, x)


def _is_dry(profile: ReactiveProfile, t_h: float) -> bool:
    if profile.cycle_h is None:
        return False
    if profile.cycle_h <= 0:
        raise ValueError("cycle_h must be positive when enabled")
    phase = (float(t_h) % profile.cycle_h) / profile.cycle_h
    return phase < profile.dry_fraction


def step_reactive_state(
    state: ReactiveState,
    profile: ReactiveProfile,
    *,
    t_h: float,
) -> None:
    """Advance one conservative coarse-grained chemistry step."""
    before = state.total_units()
    dt = float(profile.dt_h)

    U = _finite_nonnegative(state.unreactive_free, name="unreactive_free")
    A = _finite_nonnegative(state.reactive_free, name="reactive_free")
    C = np.asarray(state.chains, dtype=np.float64).copy()

    if not np.isfinite(C).all() or np.any(C < -1e-12):
        raise FloatingPointError("chain state invalid before update")
    C = np.maximum(C, 0.0)

    # Reactive-state exchange.
    activate = min(U, max(0.0, profile.k_activation * U * dt))
    deactivate = min(A, max(0.0, profile.k_deactivation * A * dt))
    U = U - activate + deactivate
    A = A + activate - deactivate

    dry = _is_dry(profile, t_h)
    k_ext = profile.k_extension * (
        profile.dry_extension_multiplier if dry else 1.0
    )
    k_hyd = profile.k_hydrolysis * (
        profile.dry_hydrolysis_multiplier if dry else 1.0
    )

    # Requests from chain source bins. Extension and hydrolysis compete for the
    # same source chain population and are jointly capped.
    ext = np.zeros_like(C)
    hyd = np.zeros_like(C)
    for length in range(2, len(C) - 1):
        ext[length] = max(0.0, k_ext * A * C[length] * dt)
    for length in range(2, len(C)):
        hyd[length] = max(
            0.0,
            k_hyd * max(1, length - 1) * C[length] * dt,
        )

    requested_chain_out = ext + hyd
    positive = requested_chain_out > 0.0
    source_scale = np.ones_like(C)
    source_scale[positive] = np.minimum(
        1.0,
        C[positive] / requested_chain_out[positive],
    )
    ext *= source_scale
    hyd *= source_scale

    # Nucleation competes with extension for the same reactive free pool.
    nucleation = max(0.0, profile.k_nucleation * A * A * dt)
    reactive_use = 2.0 * nucleation + float(np.sum(ext))
    if reactive_use > A and reactive_use > 0.0:
        scale = A / reactive_use
        nucleation *= scale
        ext *= scale

    delta = np.zeros_like(C)
    delta[2] += nucleation
    A -= 2.0 * nucleation

    for length in range(2, len(C) - 1):
        flux = float(ext[length])
        if flux <= 0.0:
            continue
        delta[length] -= flux
        delta[length + 1] += flux
        A -= flux

    for length in range(2, len(C)):
        flux = float(hyd[length])
        if flux <= 0.0:
            continue
        delta[length] -= flux
        if length == 2:
            U += 2.0 * flux
        else:
            delta[length - 1] += flux
            U += flux

    C += delta
    if not np.isfinite(C).all() or np.any(C < -1e-10):
        raise FloatingPointError("reactive-state update produced invalid chains")
    C = np.maximum(C, 0.0)
    U = _finite_nonnegative(U, name="unreactive_free_after")
    A = _finite_nonnegative(A, name="reactive_free_after")

    state.unreactive_free = U
    state.reactive_free = A
    state.chains = C

    after = state.total_units()
    rel = abs(after - before) / max(1.0, abs(before))
    state.max_relative_conservation_error = max(
        state.max_relative_conservation_error,
        rel,
    )
    if rel > 1e-10:
        raise FloatingPointError(
            "reactive-state nucleotide conservation failed: "
            f"before={before:.16g}, after={after:.16g}, rel={rel:.3e}"
        )


def simulate_reactive_profile(profile: ReactiveProfile) -> dict[str, object]:
    if profile.hours <= 0 or profile.dt_h <= 0:
        raise ValueError("hours and dt_h must be positive")
    if profile.max_len < 50:
        raise ValueError("max_len must be >=50 for v0.3 calibration")

    state = ReactiveState.initialize(profile)
    n_steps = int(round(profile.hours / profile.dt_h))
    for step in range(n_steps):
        step_reactive_state(
            state,
            profile,
            t_h=step * profile.dt_h,
        )

    if not np.isfinite(state.chains).all():
        raise FloatingPointError("non-finite final chain state")

    total = state.total_units()
    chain_units = float(
        np.dot(np.arange(len(state.chains), dtype=float), state.chains)
    )
    return {
        "profile": profile.name,
        "hours": float(profile.hours),
        "total_units": total,
        "unreactive_free": float(state.unreactive_free),
        "reactive_free": float(state.reactive_free),
        "chain_units": chain_units,
        "mass_fraction_ge_10": state.mass_fraction_ge(10),
        "mass_fraction_ge_30": state.mass_fraction_ge(30),
        "mass_fraction_ge_40": state.mass_fraction_ge(40),
        "max_length_mass_fraction_ge_1e_10": state.max_length_at_mass_fraction(1e-10),
        "max_relative_conservation_error": float(
            state.max_relative_conservation_error
        ),
        "geometry_used": False,
        "zeta_used": False,
        "replication_used": False,
    }


def run_reactive_state_calibration() -> tuple[pd.DataFrame, dict[str, object]]:
    cyclic = simulate_reactive_profile(CYCLIC_WET_DRY_PROFILE)
    clay = simulate_reactive_profile(ACTIVATED_CLAY_PROFILE)
    rows = pd.DataFrame([cyclic, clay])

    cyclic_ok = bool(
        3e-4 <= float(cyclic["mass_fraction_ge_10"]) <= 3e-3
    )
    clay_mass_ok = bool(
        1e-9 <= float(clay["mass_fraction_ge_40"]) <= 1e-6
    )
    clay_tail = int(clay["max_length_mass_fraction_ge_1e_10"])
    clay_tail_ok = bool(40 <= clay_tail <= 55)
    conservation_ok = bool(
        float(rows["max_relative_conservation_error"].max()) <= 1e-10
        and np.allclose(
            rows["total_units"].to_numpy(dtype=float),
            10_000.0,
            rtol=1e-10,
            atol=1e-8,
        )
    )
    boundary_ok = bool(
        (~rows["geometry_used"]).all()
        and (~rows["zeta_used"]).all()
        and (~rows["replication_used"]).all()
    )

    passed = bool(
        cyclic_ok
        and clay_mass_ok
        and clay_tail_ok
        and conservation_ok
        and boundary_ok
    )
    verdict = (
        "PASS_REACTIVE_STATE_CALIBRATION"
        if passed
        else "FAIL_REACTIVE_STATE_CALIBRATION"
    )
    summary = {
        "schema": "ORIGINS_FIRST_RNA_REACTIVE_STATE_V0_3",
        "status": verdict,
        "calibration_not_prediction": True,
        "cyclic_mass_fraction_ge_10": float(cyclic["mass_fraction_ge_10"]),
        "activated_clay_mass_fraction_ge_40": float(clay["mass_fraction_ge_40"]),
        "activated_clay_tail_length_ge_1e_10": clay_tail,
        "max_relative_conservation_error": float(
            rows["max_relative_conservation_error"].max()
        ),
        "cyclic_anchor_pass": cyclic_ok,
        "activated_clay_mass_anchor_pass": clay_mass_ok,
        "activated_clay_tail_anchor_pass": clay_tail_ok,
        "conservation_pass": conservation_ok,
        "causal_boundary_pass": boundary_ok,
        "geometry_used": False,
        "zeta_used": False,
        "replication_used": False,
    }
    return rows, summary
