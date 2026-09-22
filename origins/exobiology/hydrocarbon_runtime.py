from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import label

from .profiles import HYDROCARBON_CANDIDATE, WorldEnvironment


RUNTIME_CODE = "HYDROCARBON_CANDIDATE_V0_1"
PARAMETER_STATUS = "UNVALIDATED_DIMENSIONLESS_CANDIDATE"
PHYSICAL_BINDING = "OPEN"


@dataclass(frozen=True)
class HydrocarbonCandidateParameters:
    """Exploratory dimensionless parameters for a methane/ethane candidate.

    Values are computational test parameters, not measured Titan-biochemistry
    constants.
    """

    dt: float = 0.02
    dissolved_diffusion: float = 0.035
    interface_diffusion: float = 0.002
    information_diffusion: float = 0.001
    boundary_diffusion: float = 0.0005
    energy_diffusion: float = 0.06
    energy_input: float = 0.006
    energy_loss: float = 0.004
    interface_capture_rate: float = 0.020
    interface_release_rate: float = 0.004
    information_assembly_rate: float = 0.008
    boundary_assembly_rate: float = 0.006
    information_decay_rate: float = 0.0015
    boundary_decay_rate: float = 0.0008
    selection_strength: float = 0.45
    inheritance_rate: float = 0.25
    mutation_sigma: float = 0.008
    information_threshold: float = 0.04
    boundary_threshold: float = 0.04

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


class HydrocarbonCandidateSimulator:
    """Dedicated non-RNA, non-lipid hydrocarbon candidate runtime.

    Functional state:
      S : dissolved/non-polar precursor material
      A : interfacial/aggregate-bound precursor reservoir
      I : persistent information-carrier proxy
      B : persistent boundary/compartment proxy
      E : environmental energy-availability field
      Q : heritable information-state trait

    S + A + I + B is conserved. The explicit A reservoir distinguishes this
    model from the ammonia candidate and allows interface-mediated assembly
    without assuming a lipid membrane or a specific azotosome chemistry.
    """

    def __init__(
        self,
        scenario,
        *,
        Nx: int = 48,
        Ny: int = 48,
        parameters: HydrocarbonCandidateParameters | None = None,
        seed: int | None = None,
    ):
        self.environment = WorldEnvironment.from_scenario(scenario)
        HYDROCARBON_CANDIDATE.assert_environment_compatible(self.environment)
        if getattr(scenario, "biochemistry_profile", None) != HYDROCARBON_CANDIDATE.code:
            raise ValueError("scenario must bind HYDROCARBON_CANDIDATE")

        self.scenario = scenario
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        if self.Nx <= 1 or self.Ny <= 1:
            raise ValueError("grid dimensions must be > 1")

        self.parameters = parameters or HydrocarbonCandidateParameters()
        self.parameters.validate()
        self._rng = np.random.default_rng(scenario.seed if seed is None else seed)

        self.S: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.I: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.E: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.interface_template: np.ndarray | None = None

        self.time = 0.0
        self.energy_throughput_integral = 0.0
        self.material_reference: float | None = None

    def initialize(self) -> None:
        rng = self._rng
        self.S = rng.uniform(0.35, 0.65, (self.Nx, self.Ny)).astype(float)
        self.A = np.zeros((self.Nx, self.Ny), dtype=float)
        self.I = np.zeros((self.Nx, self.Ny), dtype=float)
        self.B = np.zeros((self.Nx, self.Ny), dtype=float)
        self.E = rng.uniform(0.02, 0.08, (self.Nx, self.Ny)).astype(float)
        self.Q = np.zeros((self.Nx, self.Ny), dtype=float)

        # Frozen heterogeneous interface propensity. This is a model geometry,
        # not a molecular claim.
        raw = rng.uniform(0.0, 1.0, (self.Nx, self.Ny))
        self.interface_template = (
            raw
            + np.roll(raw, 1, axis=0)
            + np.roll(raw, -1, axis=0)
            + np.roll(raw, 1, axis=1)
            + np.roll(raw, -1, axis=1)
        ) / 5.0
        self.interface_template /= max(float(np.max(self.interface_template)), 1e-12)

        self.time = 0.0
        self.energy_throughput_integral = 0.0
        self.material_reference = self.material_total()
        self._validate_state()

    def _require_initialized(self) -> None:
        if any(x is None for x in (self.S, self.A, self.I, self.B, self.E, self.Q, self.interface_template)):
            raise RuntimeError("hydrocarbon candidate runtime is not initialized")

    def material_total(self) -> float:
        self._require_initialized()
        return float(np.sum(self.S) + np.sum(self.A) + np.sum(self.I) + np.sum(self.B))

    def _validate_state(self) -> None:
        self._require_initialized()
        for name in ("S", "A", "I", "B", "E", "Q"):
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

    def step_interface_exchange(self) -> None:
        """Conservative S <-> A exchange controlled by interface propensity."""
        self._require_initialized()
        p = self.parameters
        capture = np.minimum(
            p.interface_capture_rate * self.S * self.interface_template * p.dt,
            self.S,
        )
        release = np.minimum(
            p.interface_release_rate * self.A * (1.0 - self.interface_template) * p.dt,
            self.A,
        )
        self.S += -capture + release
        self.A += capture - release

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

    def step_information_and_selection(self) -> None:
        """Interface-fed information proxy with inheritance and selection."""
        self._require_initialized()
        p = self.parameters
        inherited = self._neighbor_parent_trait()
        mutation = self._rng.normal(0.0, p.mutation_sigma, self.Q.shape)
        candidate_trait = np.clip(inherited + mutation, 0.0, 1.0)

        selection = 1.0 + p.selection_strength * self.Q
        request = (
            p.information_assembly_rate
            * self.A
            * self.E
            * selection
            * p.dt
        )
        growth = np.minimum(request, self.A)
        old_mass = self.I.copy()
        self.A -= growth
        self.I += growth

        trait_mass = (
            old_mass * self.Q
            + growth * (
                (1.0 - p.inheritance_rate) * self.Q
                + p.inheritance_rate * candidate_trait
            )
        )
        self.Q = np.divide(
            trait_mass,
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
        self.A += decay
        np.clip(self.Q, 0.0, 1.0, out=self.Q)

    def step_boundary(self) -> None:
        """Build a generic persistent boundary from interfacial material."""
        self._require_initialized()
        p = self.parameters
        request = (
            p.boundary_assembly_rate
            * self.A
            * self.E
            * self.interface_template
            * p.dt
        )
        growth = np.minimum(request, self.A)
        self.A -= growth
        self.B += growth

        decay = np.minimum(p.boundary_decay_rate * self.B * p.dt, self.B)
        self.B -= decay
        self.A += decay

    def step_transport(self) -> None:
        self._require_initialized()
        p = self.parameters
        self.S = _conservative_diffuse(self.S, p.dissolved_diffusion, p.dt)
        self.A = _conservative_diffuse(self.A, p.interface_diffusion, p.dt)
        self.I = _conservative_diffuse(self.I, p.information_diffusion, p.dt)
        self.B = _conservative_diffuse(self.B, p.boundary_diffusion, p.dt)

    def step(self) -> None:
        self._require_initialized()
        before = self.material_total()
        self.step_energy_throughput()
        self.step_interface_exchange()
        self.step_information_and_selection()
        self.step_boundary()
        self.step_transport()
        self._validate_state()
        after = self.material_total()
        tolerance = 2e-10 * max(1.0, abs(before))
        if abs(after - before) > tolerance:
            raise FloatingPointError(
                f"hydrocarbon candidate material conservation failed: "
                f"before={before:.12g}, after={after:.12g}"
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
                "observable": "generic B threshold + connected component",
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
                "observable": "interface-fed Q inheritance with mutation",
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
            "profile": HYDROCARBON_CANDIDATE.code,
            "epistemic_status": "CANDIDATE",
            "parameter_status": PARAMETER_STATUS,
            "physical_binding": PHYSICAL_BINDING,
            "dedicated_non_rna_runtime": True,
            "dedicated_non_lipid_runtime": True,
            "explicit_interface_reservoir": True,
            "specific_azotosome_claim": False,
            "exotic_biology_established": False,
            "interpretation_allowed": "COMPUTATIONAL_CANDIDATE_ONLY",
        }
