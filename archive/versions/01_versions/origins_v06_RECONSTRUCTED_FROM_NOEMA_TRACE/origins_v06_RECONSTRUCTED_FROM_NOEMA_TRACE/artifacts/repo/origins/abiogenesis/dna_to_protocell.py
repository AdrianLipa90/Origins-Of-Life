"""DNA-four-block to protocell transition model for Origins/NOEMA v0.6.

Boundary
--------
This is a deterministic relational proxy model. It does not claim that DNA
preceded RNA in empirical abiogenesis, does not simulate quantum chemistry,
and does not infer a real protocell from geological data. It asks a narrower
question: given the v0.5 four-block DNA relation invariant, how does a
time-varying climate-holonomic environment change the readiness of that
information scaffold to become co-localised with a compartment/protocell
layer?

The word "DNA" here denotes the four-base relation grammar A/C/G/T as a
stable information alphabet proxy. The protocell side is a compartmental
proxy: membrane availability, co-localisation pressure, cycling, mineral
templating and recurrence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .dna_four_blocks import compute_four_block_relation

SCHEMA = "ORIGINS_NOEMA_DNA_TO_PROTOCELL_V0_6"


def _clamp01(x: float) -> float:
    if not isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


@dataclass(frozen=True)
class ClimateHolonomicStage:
    """A relative-time climate/holonomy vector.

    All fields are unitless proxies in [0, 1]. They are not geological
    measurements. The model uses them to represent how changing conditions
    modulate concentration, cycling, templating, compartment availability
    and recurrence through time.
    """

    name: str
    relative_time_order: int
    hydration: float
    thermal_gradient: float
    wet_dry_cycling: float
    mineral_template: float
    amphiphile_availability: float
    ionic_moderation: float
    uv_stress: float
    compartment_pressure: float
    holonomic_recurrence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProtocellStageScore:
    stage: ClimateHolonomicStage
    information_core_proxy: float
    climate_holonomy_proxy: float
    polymer_persistence_proxy: float
    compartment_co_localisation_proxy: float
    coupling_proxy: float
    protocell_readiness_proxy: float
    threshold: float
    crosses_threshold: bool
    transition_label: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.to_dict()
        return data


@dataclass(frozen=True)
class DNAToProtocellReport:
    schema: str
    source_four_block_schema: str
    threshold_derivation: Dict[str, float]
    stages: List[ProtocellStageScore]
    first_transition_stage: Optional[str]
    first_transition_index: Optional[int]
    metrics: Dict[str, float]
    interpretation: str
    certification_boundary: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "source_four_block_schema": self.source_four_block_schema,
            "threshold_derivation": dict(self.threshold_derivation),
            "stages": [s.to_dict() for s in self.stages],
            "first_transition_stage": self.first_transition_stage,
            "first_transition_index": self.first_transition_index,
            "metrics": dict(self.metrics),
            "interpretation": self.interpretation,
            "certification_boundary": dict(self.certification_boundary),
        }

    def to_noema_candidate(self) -> Dict[str, Any]:
        return {
            "schema": "NOEMA_INGEST_CANDIDATE_V0_1",
            "namespace": "Origins-Of-Life",
            "project_id": "Origins-Of-Life::dna_to_protocell_v06",
            "layer": "episodic",
            "snap_type": "iteration_state",
            "tags": [
                "origins",
                "dna-four-blocks",
                "protocell",
                "climate-holonomy",
                "semantic-compression",
            ],
            "meaning_candidate": {
                "hypothesis": self.interpretation,
                "confidence": round(self.metrics.get("peak_protocell_readiness_proxy", 0.0), 6),
                "source": "local NOEMA v05b + deterministic v06 proxy model",
                "context_required": True,
                "canon_allowed": False,
                "needs_repair": False,
                "uncertainty_operator": "chyba",
            },
            "payload": self.to_dict(),
        }


def default_climate_holonomic_stages() -> List[ClimateHolonomicStage]:
    """Return a relative climate-holonomic trajectory.

    The stages are ordered states, not calendar-dated geological claims.
    They encode a plausible model trajectory from information alphabet only,
    through cycling and surface concentration, toward compartmentalisation.
    """

    return [
        ClimateHolonomicStage(
            name="alphabet_only_open_surface",
            relative_time_order=0,
            hydration=0.35,
            thermal_gradient=0.30,
            wet_dry_cycling=0.20,
            mineral_template=0.40,
            amphiphile_availability=0.10,
            ionic_moderation=0.30,
            uv_stress=0.70,
            compartment_pressure=0.10,
            holonomic_recurrence=0.25,
        ),
        ClimateHolonomicStage(
            name="surface_cycling_concentration",
            relative_time_order=1,
            hydration=0.55,
            thermal_gradient=0.65,
            wet_dry_cycling=0.80,
            mineral_template=0.75,
            amphiphile_availability=0.35,
            ionic_moderation=0.45,
            uv_stress=0.65,
            compartment_pressure=0.35,
            holonomic_recurrence=0.55,
        ),
        ClimateHolonomicStage(
            name="vesicle_contact_with_polymer_trace",
            relative_time_order=2,
            hydration=0.65,
            thermal_gradient=0.55,
            wet_dry_cycling=0.70,
            mineral_template=0.65,
            amphiphile_availability=0.70,
            ionic_moderation=0.60,
            uv_stress=0.45,
            compartment_pressure=0.75,
            holonomic_recurrence=0.70,
        ),
        ClimateHolonomicStage(
            name="encapsulated_recurrence",
            relative_time_order=3,
            hydration=0.75,
            thermal_gradient=0.45,
            wet_dry_cycling=0.55,
            mineral_template=0.55,
            amphiphile_availability=0.85,
            ionic_moderation=0.75,
            uv_stress=0.35,
            compartment_pressure=0.90,
            holonomic_recurrence=0.85,
        ),
        ClimateHolonomicStage(
            name="protocell_candidate",
            relative_time_order=4,
            hydration=0.80,
            thermal_gradient=0.40,
            wet_dry_cycling=0.50,
            mineral_template=0.50,
            amphiphile_availability=0.90,
            ionic_moderation=0.85,
            uv_stress=0.25,
            compartment_pressure=0.95,
            holonomic_recurrence=0.95,
        ),
    ]


def compute_information_core_proxy(four_block_metrics: Mapping[str, float]) -> float:
    """Compute the information-scaffold core from v0.5 metrics."""

    keys = [
        "complement_selectivity_proxy",
        "grammar_compression_proxy",
        "identity_retention_proxy",
        "transition_readiness_proxy",
        "alphabet_closure_proxy",
        "class_symmetry_proxy",
    ]
    return _clamp01(sum(float(four_block_metrics[k]) for k in keys) / len(keys))


def climate_holonomy_score(stage: ClimateHolonomicStage) -> float:
    """Score environmental cycling/recurrence support.

    UV stress is not treated as monotonic good or bad; the proxy rewards a
    moderate stress point near 0.45 and penalises extremes.
    """

    uv_balance = _clamp01(1.0 - abs(stage.uv_stress - 0.45))
    return _clamp01(
        (
            stage.hydration
            + stage.thermal_gradient
            + stage.wet_dry_cycling
            + stage.mineral_template
            + stage.ionic_moderation
            + uv_balance
            + stage.holonomic_recurrence
        )
        / 7.0
    )


def polymer_persistence_score(stage: ClimateHolonomicStage) -> float:
    uv_balance = _clamp01(1.0 - abs(stage.uv_stress - 0.45))
    return _clamp01(
        0.35 * stage.hydration
        + 0.30 * stage.mineral_template
        + 0.20 * stage.ionic_moderation
        + 0.15 * uv_balance
    )


def compartment_co_localisation_score(stage: ClimateHolonomicStage) -> float:
    return _clamp01(
        0.50 * stage.amphiphile_availability
        + 0.20 * stage.compartment_pressure
        + 0.15 * stage.ionic_moderation
        + 0.15 * stage.wet_dry_cycling
    )


def score_stage(
    stage: ClimateHolonomicStage,
    information_core_proxy: float,
    threshold: float,
) -> ProtocellStageScore:
    climate = climate_holonomy_score(stage)
    polymer = polymer_persistence_score(stage)
    compartment = compartment_co_localisation_score(stage)
    coupling = _clamp01((information_core_proxy * climate * polymer * compartment) ** 0.25)
    readiness = _clamp01(
        information_core_proxy
        * (0.25 * climate + 0.25 * polymer + 0.25 * compartment + 0.25 * coupling)
    )
    crosses = readiness >= threshold
    return ProtocellStageScore(
        stage=stage,
        information_core_proxy=round(information_core_proxy, 6),
        climate_holonomy_proxy=round(climate, 6),
        polymer_persistence_proxy=round(polymer, 6),
        compartment_co_localisation_proxy=round(compartment, 6),
        coupling_proxy=round(coupling, 6),
        protocell_readiness_proxy=round(readiness, 6),
        threshold=round(threshold, 6),
        crosses_threshold=bool(crosses),
        transition_label="protocell_candidate_crossing" if crosses else "pre_protocell",
    )


def compute_dna_to_protocell_transition(
    stages: Optional[Sequence[ClimateHolonomicStage]] = None,
) -> DNAToProtocellReport:
    """Compute v0.6 DNA-four-block -> protocell readiness trajectory."""

    four_block = compute_four_block_relation()
    metrics = four_block.metrics
    information_core = compute_information_core_proxy(metrics)
    # Derived threshold: v0.5 grammar compression times complement selectivity.
    # This requires both alphabet reducibility and relation selectivity.
    threshold = _clamp01(
        float(metrics["grammar_compression_proxy"])
        * float(metrics["complement_selectivity_proxy"])
    )
    input_stages = list(stages) if stages is not None else default_climate_holonomic_stages()
    scores = [score_stage(s, information_core, threshold) for s in input_stages]
    first = next((s for s in scores if s.crosses_threshold), None)
    peak = max(s.protocell_readiness_proxy for s in scores) if scores else 0.0
    start = scores[0].protocell_readiness_proxy if scores else 0.0
    improvement = _clamp01(peak - start)
    metrics_out = {
        "stage_count": float(len(scores)),
        "information_core_proxy": round(information_core, 6),
        "threshold_proxy": round(threshold, 6),
        "peak_protocell_readiness_proxy": round(peak, 6),
        "initial_readiness_proxy": round(start, 6),
        "readiness_gain_proxy": round(improvement, 6),
        "transition_detected": 1.0 if first else 0.0,
        "transition_index": float(first.stage.relative_time_order) if first else -1.0,
        "final_climate_holonomy_proxy": scores[-1].climate_holonomy_proxy if scores else 0.0,
        "final_compartment_co_localisation_proxy": scores[-1].compartment_co_localisation_proxy if scores else 0.0,
    }
    interpretation = (
        "The v0.6 model says that the four-block DNA relation invariant alone is not a protocell. "
        "It becomes a protocell candidate only when the invariant information scaffold is repeatedly coupled "
        "to a time-varying climate-holonomic environment that provides cycling, mineral/ionic persistence, "
        "amphiphile-supported compartment co-localisation and recurrence. The first crossing is a proxy for "
        "co-localised proto-memory in a compartment, not empirical proof of abiogenesis."
    )
    return DNAToProtocellReport(
        schema=SCHEMA,
        source_four_block_schema=four_block.schema,
        threshold_derivation={
            "grammar_compression_proxy": float(metrics["grammar_compression_proxy"]),
            "complement_selectivity_proxy": float(metrics["complement_selectivity_proxy"]),
            "threshold_proxy": round(threshold, 6),
        },
        stages=scores,
        first_transition_stage=first.stage.name if first else None,
        first_transition_index=first.stage.relative_time_order if first else None,
        metrics=metrics_out,
        interpretation=interpretation,
        certification_boundary={
            "semantic_relation_model_claimed": True,
            "time_varying_climate_holonomy_claimed": True,
            "protocell_readiness_proxy_claimed": True,
            "empirical_abiogenesis_proof_claimed": False,
            "geological_timeline_claimed": False,
            "dna_first_historical_claimed": False,
            "quantum_chemistry_claimed": False,
            "thermodynamic_membrane_simulation_claimed": False,
        },
    )


def run_dna_to_protocell_demo(as_noema: bool = False) -> Dict[str, Any] | DNAToProtocellReport:
    report = compute_dna_to_protocell_transition()
    return report.to_noema_candidate() if as_noema else report
