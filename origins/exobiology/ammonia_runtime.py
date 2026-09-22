from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import label

from .profiles import AMMONIA_CANDIDATE, WorldEnvironment


RUNTIME_CODE = "AMMONIA_CANDIDATE_V0_1"
PARAMETER_STATUS = "UNVALIDATED_DIMENSIONLESS_CANDIDATE"
PHYSICAL_BINDING = "OPEN"


@dataclass(frozen=True)
class AmmoniaCandidateParameters:
    """Dimensionless exploratory parameters for the NH3 candidate runtime.

    These values define a falsifiable computational candidate. They are not
    measured ammonia-biochemistry constants.
    """

    dt: float = 0.02
    precursor_diffusion: float = 0.08
    energy_diffusion: float = 0.10
    information_diffusion: float = 0.003
    boundary_diffusion: float = 0.001
    energy_input: float = 0.015
    energy_loss: float = 0.010
    information_assembly_rate: float = 0.018
    boundary_assembly_rate: float = 0.010
    information_decay_rate: float = 0.004
    boundary_decay_rate: float = 0.002
    selection_strength: float = 0.50
    inheritance_rate: float = 0.30
    mutation_sigma: float = 0.01
    information_threshold: float = 0.08
    boundary_threshold: float = 0.08

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.selection_strength > 1.0:
            raise ValueError("selection_strength must be <= 1")
        if self.inheritance_rate > 1.0:
            raise ValueError("inheritance_rate must be <= 1")


def _laplacian(z: np.ndarray) -> np.ndarray:
    return (
        -4.0 * z
        + np.roll(z, 1, axis=0)
        + np.roll(z, -1, axis=0)
        + np.roll(z, 1, axis=1)
        + np.roll(z, -1, axis=1)
    )


def _conservative_diffuse(field: np.ndarray, coefficient: float, dt: float) -> np.ndarray:
    out = field + coefficient * dt * _laplacian(field)
    if not np.isfinite(out).all():
        raise FloatingPointError("diffusion produced NaN/Inf")
    if float(np.min(out)) < -1e-10:
        raise FloatingPointError(
            "diffusion left the positivity-stable regime; reduce dt or coefficient"
        )
    return np.maximum(out, 0.0)


class AmmoniaCandidateSimulator:
    """Dedicated non-RNA, non-lipid NH3 candidate-life runtime.

    State variables are functional proxies:
      P : free material / building-block pool
      I : persistent information-carrier proxy
      B : compartment-boundary material proxy
      E : externally driven energy-availability field
      Q : heritable information-state trait in [0, 1]

    P + I + B is conserved apart from numerical tolerance. E is intentionally
    open because it represents environmental energy throughput.

    This is a computational candidate with OPEN physical chemistry, not a claim
    that these proxy species exist in ammonia solvent.
    """

    def __init__(
        self,
        scenario,
        *,
        Nx: int = 48,
        Ny: int = 48,
        parameters: AmmoniaCandidateParameters | None = None,
        seed: int | None = None,
    ):
        self.environment = WorldEnvironment.from_scenario(scenario)
        AMMONIA_CANDIDATE.assert_environment_compatible(self.environment)
        if getattr(scenario, "biochemistry_profile", None) != AMMONIA_CANDIDATE.code:
            raise ValueError("scenario must bind AMMONIA_CANDIDATE")

        self.scenario = scenario
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        if self.Nx <= 1 or self.Ny <= 1:
            raise ValueError("grid dimensions must be > 1")

        self.parameters = parameters or AmmoniaCandidateParameters()
        self.parameters.validate()
        self._rng = np.random.default_rng(scenario.seed if seed is None else seed)

        self.P: np.ndarray | None = None
        self.I: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.E: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.time = 0.0
        self.energy_throughput_integral = 0.0
        self.material_reference: float | None = None

    def initialize(self) -> None:
        rng = self._rng
        self.P = rng.uniform(0.35, 0.65, (self.Nx, self.Ny)).astype(float)
        self.I = np.zeros((self.Nx, self.Ny), dtype=float)
        self.B = np.zeros((self.Nx, self.Ny), dtype=float)
        self.E = rng.uniform(0.05, 0.15, (self.Nx, self.Ny)).astype(float)
        self.Q = np.zeros((self.Nx, self.Ny), dtype=float)
        self.time = 0.0
        self.energy_throughput_integral = 0.0
        self.material_reference = self.material_total()
        self._validate_state()

    def _require_initialized(self) -> None:
        if any(x is None for x in (self.P, self.I, self.B, self.E, self.Q)):
            raise RuntimeError("ammonia candidate runtime is not initialized")

    def material_total(self) -> float:
        self._require_initialized()
        return float(np.sum(self.P) + np.sum(self.I) + np.sum(self.B))

    def _validate_state(self) -> None:
        self._require_initialized()
        for name in ("P", "I", "B", "E", "Q"):
            field = getattr(self, name)
            if not np.isfinite(field).all():
                raise FloatingPointError(f"{name} contains NaN/Inf")
            if float(np.min(field)) < -1e-10:
                raise FloatingPointError(f"{name} contains negative state")
        if float(np.min(self.Q)) < 0.0 or float(np.max(self.Q)) > 1.0:
            raise FloatingPointError("Q escaped [0, 1]")

    def step_energy_throughput(self) -> None:
        self._require_initialized()
        p = self.parameters
        before = float(np.sum(self.E))
        self.E = _conservative_diffuse(self.E, p.energy_diffusion, p.dt)
        self.E += p.energy_input * p.dt
        self.E *= math.exp(-p.energy_loss * p.dt)
        np.maximum(self.E, 0.0, out=self.E)
        after = float(np.sum(self.E))
        self.energy_throughput_integral += max(0.0, after - before)

    def _neighbor_parent_trait(self) -> np.ndarray:
        weighted = (
            np.roll(self.I * self.Q, 1, axis=0)
            + np.roll(self.I * self.Q, -1, axis=0)
            + np.roll(self.I * self.Q, 1, axis=1)
            + np.roll(self.I * self.Q, -1, axis=1)
        )
        mass = (
            np.roll(self.I, 1, axis=0)
            + np.roll(self.I, -1, axis=0)
            + np.roll(self.I, 1, axis=1)
            + np.roll(self.I, -1, axis=1)
        )
        return np.divide(weighted, mass, out=np.zeros_like(weighted), where=mass > 1e-15)

    def step_information_and_inheritance(self) -> None:
        self._require_initialized()
        p = self.parameters

        inherited = self._neighbor_parent_trait()
        mutation = self._rng.normal(0.0, p.mutation_sigma, self.Q.shape)
        candidate_trait = np.clip(inherited + mutation, 0.0, 1.0)

        selection = 1.0 + p.selection_strength * self.Q
        request = (
            p.information_assembly_rate
            * self.P
            * self.E
            * selection
            * p.dt
        )
        growth = np.minimum(request, self.P)

        old_mass = self.I.copy()
        self.P -= growth
        self.I += growth

        new_trait = (
            (old_mass * self.Q)
            + (growth * ((1.0 - p.inheritance_rate) * self.Q + p.inheritance_rate * candidate_trait))
        )
        self.Q = np.divide(
            new_trait,
            self.I,
            out=np.zeros_like(self.Q),
            where=self.I > 1e-15,
        )

        decay_modifier = 1.0 - p.selection_strength * self.Q
        decay = np.minimum(
            p.information_decay_rate * self.I * decay_modifier * p.dt,
            self.I,
        )
        self.I -= decay
        self.P += decay
        np.clip(self.Q, 0.0, 1.0, out=self.Q)

    def step_boundary(self) -> None:
        self._require_initialized()
        p = self.parameters
        request = p.boundary_assembly_rate * self.P * self.E * p.dt
        growth = np.minimum(request, self.P)
        self.P -= growth
        self.B += growth

        decay = np.minimum(p.boundary_decay_rate * self.B * p.dt, self.B)
        self.B -= decay
        self.P += decay

    def step_transport(self) -> None:
        self._require_initialized()
        p = self.parameters
        self.P = _conservative_diffuse(self.P, p.precursor_diffusion, p.dt)
        self.I = _conservative_diffuse(self.I, p.information_diffusion, p.dt)
        self.B = _conservative_diffuse(self.B, p.boundary_diffusion, p.dt)

    def step(self) -> None:
        self._require_initialized()
        before = self.material_total()
        self.step_energy_throughput()
        self.step_information_and_inheritance()
        self.step_boundary()
        self.step_transport()
        self._validate_state()
        after = self.material_total()
        tolerance = 2e-10 * max(1.0, abs(before))
        if abs(after - before) > tolerance:
            raise FloatingPointError(
                f"NH3 candidate material conservation failed: before={before:.12g}, after={after:.12g}"
            )
        self.time += self.parameters.dt

    def run(self, steps: int) -> None:
        if int(steps) < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(int(steps)):
            self.step()

    def candidate_compartments(self) -> dict[str, object]:
        self._require_initialized()
        p = self.parameters
        mask = (self.B >= p.boundary_threshold) & (self.I >= p.information_threshold)
        labels, count = label(mask)
        return {
            "count": int(count),
            "area_pixels": int(mask.sum()),
            "labels": labels,
        }

    def life_invariant_status(self) -> dict[str, dict[str, object]]:
        self._require_initialized()
        compartments = self.candidate_compartments()
        return {
            "BOUNDED_SYSTEM": {
                "operationalized": True,
                "observable": "B threshold + connected component",
                "physical_binding": PHYSICAL_BINDING,
                "detected_count": compartments["count"],
            },
            "ENERGY_THROUGHPUT": {
                "operationalized": True,
                "observable": "external E input and loss",
                "physical_binding": PHYSICAL_BINDING,
                "throughput_integral": float(self.energy_throughput_integral),
            },
            "PERSISTENT_INFORMATION_STATE": {
                "operationalized": True,
                "observable": "I field with Q state",
                "physical_binding": PHYSICAL_BINDING,
                "information_mass": float(np.sum(self.I)),
            },
            "HERITABLE_STATE_TRANSFORMATION": {
                "operationalized": True,
                "observable": "neighbor-weighted Q inheritance with mutation",
                "physical_binding": PHYSICAL_BINDING,
                "trait_mean": float(np.mean(self.Q)),
            },
            "SELECTION_OR_DIFFERENTIAL_PERSISTENCE": {
                "operationalized": True,
                "observable": "Q-dependent assembly and decay",
                "physical_binding": PHYSICAL_BINDING,
                "selection_strength": float(self.parameters.selection_strength),
            },
        }

    def claim_status(self) -> dict[str, object]:
        return {
            "runtime": RUNTIME_CODE,
            "profile": AMMONIA_CANDIDATE.code,
            "epistemic_status": "CANDIDATE",
            "parameter_status": PARAMETER_STATUS,
            "physical_binding": PHYSICAL_BINDING,
            "dedicated_non_rna_runtime": True,
            "dedicated_non_lipid_runtime": True,
            "exotic_biology_established": False,
            "interpretation_allowed": "COMPUTATIONAL_CANDIDATE_ONLY",
        }
