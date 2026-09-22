from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class EpistemicStatus(str, Enum):
    REFERENCE_MODEL = "REFERENCE_MODEL"
    CANDIDATE = "CANDIDATE"


class RuntimeStatus(str, Enum):
    REFERENCE_IMPLEMENTED = "REFERENCE_IMPLEMENTED"
    TERRACENTRIC_CONTROL_ONLY = "TERRACENTRIC_CONTROL_ONLY"


LIFE_RELATIONAL_INVARIANTS: tuple[str, ...] = (
    "BOUNDED_SYSTEM",
    "ENERGY_THROUGHPUT",
    "PERSISTENT_INFORMATION_STATE",
    "HERITABLE_STATE_TRANSFORMATION",
    "SELECTION_OR_DIFFERENTIAL_PERSISTENCE",
)


@dataclass(frozen=True)
class WorldEnvironment:
    """Physical world description, deliberately separate from biology."""

    name: str
    location: str
    solvent: str
    temp_C: float
    pressure_atm: float
    pH: float
    redox: str
    energy_source: str
    uv_flux: float

    @classmethod
    def from_scenario(cls, scenario) -> "WorldEnvironment":
        solvent = getattr(scenario.solvent, "value", scenario.solvent)
        return cls(
            name=str(scenario.name),
            location=str(scenario.location),
            solvent=str(solvent),
            temp_C=float(scenario.temp_C),
            pressure_atm=float(scenario.pressure_atm),
            pH=float(scenario.pH),
            redox=str(scenario.redox),
            energy_source=str(scenario.energy_source),
            uv_flux=float(scenario.UV_flux),
        )


@dataclass(frozen=True)
class BiochemistryProfile:
    """A biology hypothesis independent of any particular world configuration."""

    code: str
    display_name: str
    epistemic_status: EpistemicStatus
    compatible_solvents: tuple[str, ...]
    information_carrier: str
    compartment_strategy: str
    building_block_strategy: str
    energy_coupling: str
    runtime_status: RuntimeStatus
    dedicated_runtime: str | None = None
    dedicated_runtime_status: str = "NONE"
    invariant_requirements: tuple[str, ...] = LIFE_RELATIONAL_INVARIANTS
    notes: tuple[str, ...] = ()

    @property
    def runtime_implemented(self) -> bool:
        return self.runtime_status == RuntimeStatus.REFERENCE_IMPLEMENTED

    def assert_environment_compatible(self, environment: WorldEnvironment) -> None:
        if environment.solvent not in self.compatible_solvents:
            raise ValueError(
                f"biochemistry profile {self.code} is not declared compatible "
                f"with solvent {environment.solvent!r}"
            )

    def claim_status(self) -> dict[str, object]:
        return {
            "profile": self.code,
            "epistemic_status": self.epistemic_status.value,
            "runtime_status": self.runtime_status.value,
            "runtime_implemented": self.runtime_implemented,
            "dedicated_runtime": self.dedicated_runtime,
            "dedicated_runtime_status": self.dedicated_runtime_status,
            "exotic_biology_simulated": (
                self.epistemic_status == EpistemicStatus.CANDIDATE
                and self.runtime_implemented
            ),
            "life_relational_invariants": list(self.invariant_requirements),
        }


WATER_REFERENCE = BiochemistryProfile(
    code="WATER_REFERENCE",
    display_name="Water / RNA-like reference biology",
    epistemic_status=EpistemicStatus.REFERENCE_MODEL,
    compatible_solvents=("H2O",),
    information_carrier="RNA_LIKE_POLYMER_REFERENCE",
    compartment_strategy="LIPID_OR_AMPHIPHILE_BOUNDARY_REFERENCE",
    building_block_strategy="POLAR_AQUEOUS_PRECURSORS_REFERENCE",
    energy_coupling="PHOTOCHEMICAL_OR_REDOX_REFERENCE",
    runtime_status=RuntimeStatus.REFERENCE_IMPLEMENTED,
    notes=(
        "Computational reference implementation, not an empirical proof of abiogenesis.",
    ),
)

AMMONIA_CANDIDATE = BiochemistryProfile(
    code="AMMONIA_CANDIDATE",
    display_name="Ammonia-solvent candidate biology",
    epistemic_status=EpistemicStatus.CANDIDATE,
    compatible_solvents=("NH3",),
    information_carrier="HERITABLE_INFORMATION_CARRIER_OPEN",
    compartment_strategy="AMMONIA_COMPATIBLE_BOUNDARY_OPEN",
    building_block_strategy="AMMONIA_SOLVENT_BUILDING_BLOCKS_OPEN",
    energy_coupling="REDOX_OR_PHOTOCHEMICAL_COUPLING_OPEN",
    runtime_status=RuntimeStatus.TERRACENTRIC_CONTROL_ONLY,
    dedicated_runtime="AMMONIA_CANDIDATE_V0_1",
    dedicated_runtime_status="COMPUTATIONAL_CANDIDATE_IMPLEMENTED",
    notes=(
        "Dedicated non-RNA/non-lipid candidate runtime exists with OPEN physical chemistry.",
        "Current universal simulator output remains a terracentric control only.",
    ),
)

HYDROCARBON_CANDIDATE = BiochemistryProfile(
    code="HYDROCARBON_CANDIDATE",
    display_name="Methane/ethane-solvent candidate biology",
    epistemic_status=EpistemicStatus.CANDIDATE,
    compatible_solvents=("CH4/C2H6",),
    information_carrier="NONPOLAR_COMPATIBLE_HERITABLE_STATE_OPEN",
    compartment_strategy="HYDROCARBON_COMPATIBLE_BOUNDARY_OPEN",
    building_block_strategy="HYDROCARBON_SOLVENT_BUILDING_BLOCKS_OPEN",
    energy_coupling="PHOTOCHEMICAL_OR_REDOX_COUPLING_OPEN",
    runtime_status=RuntimeStatus.TERRACENTRIC_CONTROL_ONLY,
    dedicated_runtime=None,
    dedicated_runtime_status="NONE",
    notes=(
        "No dedicated hydrocarbon-biochemistry runtime is implemented yet.",
        "Current universal simulator output is a terracentric control only.",
    ),
)


_PROFILES: Mapping[str, BiochemistryProfile] = MappingProxyType(
    {
        p.code: p
        for p in (
            WATER_REFERENCE,
            AMMONIA_CANDIDATE,
            HYDROCARBON_CANDIDATE,
        )
    }
)


def get_biochemistry_profile(code: str) -> BiochemistryProfile:
    try:
        return _PROFILES[str(code)]
    except KeyError as exc:
        raise KeyError(f"unknown biochemistry profile: {code!r}") from exc


def available_biochemistry_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES)
