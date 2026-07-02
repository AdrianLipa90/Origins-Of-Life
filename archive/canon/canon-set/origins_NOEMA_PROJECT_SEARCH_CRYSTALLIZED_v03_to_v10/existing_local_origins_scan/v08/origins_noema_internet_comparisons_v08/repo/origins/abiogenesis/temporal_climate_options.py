"""Temporal climate-holonomy dynamics for Origins/NOEMA v0.7.

This layer extends v0.6 without changing its core threshold.  It asks how fast
and in what mode the DNA-four-block grammar becomes coupled to a changing
prebiotic environment.  The model remains deterministic and proxy-based:
relative time steps, not calendar dates; environmental options, not geological
reconstructions; readiness rates, not empirical proof of protocell formation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .dna_to_protocell import (
    ClimateHolonomicStage,
    DNAToProtocellReport,
    climate_holonomy_score,
    compartment_co_localisation_score,
    compute_dna_to_protocell_transition,
    default_climate_holonomic_stages,
    polymer_persistence_score,
    score_stage,
)

SCHEMA = "ORIGINS_NOEMA_TEMPORAL_CLIMATE_OPTIONS_V0_7"
CLIMATE_FIELDS = [
    "hydration",
    "thermal_gradient",
    "wet_dry_cycling",
    "mineral_template",
    "amphiphile_availability",
    "ionic_moderation",
    "uv_stress",
    "compartment_pressure",
    "holonomic_recurrence",
]


def _clamp01(x: float) -> float:
    if not isfinite(float(x)):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _round6(x: float) -> float:
    return round(float(x), 6)


@dataclass(frozen=True)
class TemporalForcingOption:
    """A deterministic relative-time forcing option.

    Field offsets are applied globally. Field slopes are multiplied by the
    normalised time coordinate in [0, 1]. Field pulses are multiplied by a
    fixed five-step wet/dry kernel [0.0, 1.0, 0.75, 0.35, 0.15] unless another
    trajectory length is used, in which case a triangular relative kernel is
    generated.
    """

    name: str
    description: str
    field_offsets: Mapping[str, float]
    field_slopes: Mapping[str, float]
    field_pulses: Mapping[str, float]
    threshold_shift: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntervalChange:
    from_stage: str
    to_stage: str
    interval_index: int
    field_deltas: Dict[str, float]
    climate_delta: float
    polymer_delta: float
    compartment_delta: float
    coupling_delta: float
    readiness_delta: float
    readiness_velocity: float
    readiness_acceleration: float
    dominant_driver: str
    change_kind: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalOptionReport:
    option: TemporalForcingOption
    transition_report: DNAToProtocellReport
    interval_changes: List[IntervalChange]
    metrics: Dict[str, float]
    classification: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option": self.option.to_dict(),
            "transition_report": self.transition_report.to_dict(),
            "interval_changes": [c.to_dict() for c in self.interval_changes],
            "metrics": dict(self.metrics),
            "classification": dict(self.classification),
        }


@dataclass(frozen=True)
class TemporalClimateOptionsReport:
    schema: str
    base_current_state_id: str
    baseline_source_schema: str
    core_v06_metrics: Dict[str, float]
    option_reports: List[TemporalOptionReport]
    best_option: str
    fastest_growth_option: str
    earliest_transition_option: Optional[str]
    interpretation: str
    certification_boundary: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "base_current_state_id": self.base_current_state_id,
            "baseline_source_schema": self.baseline_source_schema,
            "core_v06_metrics": dict(self.core_v06_metrics),
            "option_reports": [r.to_dict() for r in self.option_reports],
            "best_option": self.best_option,
            "fastest_growth_option": self.fastest_growth_option,
            "earliest_transition_option": self.earliest_transition_option,
            "interpretation": self.interpretation,
            "certification_boundary": dict(self.certification_boundary),
        }

    def to_noema_candidate(self) -> Dict[str, Any]:
        best = next((r for r in self.option_reports if r.option.name == self.best_option), None)
        confidence = best.metrics.get("peak_readiness", 0.0) if best else 0.0
        return {
            "schema": "NOEMA_INGEST_CANDIDATE_V0_1",
            "namespace": "Origins-Of-Life",
            "project_id": "Origins-Of-Life::temporal_climate_options_v07",
            "layer": "episodic",
            "snap_type": "iteration_state",
            "tags": [
                "origins",
                "dna-four-blocks",
                "protocell",
                "temporal-dynamics",
                "climate-holonomy",
                "scenario-options",
            ],
            "meaning_candidate": {
                "hypothesis": self.interpretation,
                "confidence": _round6(confidence),
                "source": "local NOEMA v06 + deterministic v07 temporal forcing model",
                "context_required": True,
                "canon_allowed": False,
                "needs_repair": False,
                "uncertainty_operator": "chyba",
            },
            "payload": self.to_dict(),
        }


def _relative_pulse(i: int, n: int) -> float:
    if n <= 1:
        return 0.0
    # Five-step wet/dry concentration kernel used by default v06 trajectory.
    fixed = [0.0, 1.0, 0.75, 0.35, 0.15]
    if n == 5:
        return fixed[i]
    t = i / (n - 1)
    return max(0.0, 1.0 - abs(t - 0.33) / 0.33) if t <= 0.66 else max(0.0, 1.0 - (t - 0.66) / 0.34) * 0.35


def apply_temporal_option(
    stages: Sequence[ClimateHolonomicStage],
    option: TemporalForcingOption,
) -> List[ClimateHolonomicStage]:
    """Return a modified stage trajectory for an environmental option."""

    n = len(stages)
    out: List[ClimateHolonomicStage] = []
    for i, stage in enumerate(stages):
        t = 0.0 if n <= 1 else i / (n - 1)
        pulse = _relative_pulse(i, n)
        values = stage.to_dict()
        for field in CLIMATE_FIELDS:
            base = float(values[field])
            offset = float(option.field_offsets.get(field, 0.0))
            slope = float(option.field_slopes.get(field, 0.0)) * t
            pulse_boost = float(option.field_pulses.get(field, 0.0)) * pulse
            values[field] = _clamp01(base + offset + slope + pulse_boost)
        values["name"] = f"{option.name}::{stage.name}"
        out.append(ClimateHolonomicStage(**values))
    return out


def classify_change_kind(readiness_delta: float, acceleration: float) -> str:
    if readiness_delta >= 0.08 and acceleration >= 0.0:
        return "accelerating_gain"
    if readiness_delta >= 0.08:
        return "fast_gain_decelerating"
    if readiness_delta >= 0.025:
        return "moderate_gain"
    if readiness_delta > -0.025:
        return "plateau_or_stabilisation"
    return "regression_or_environmental_loss"


def _dominant_driver(delta_map: Mapping[str, float]) -> str:
    if not delta_map:
        return "none"
    return max(delta_map.items(), key=lambda item: abs(item[1]))[0]


def compute_interval_changes(report: DNAToProtocellReport) -> List[IntervalChange]:
    stages = report.stages
    changes: List[IntervalChange] = []
    prev_velocity = 0.0
    for i in range(1, len(stages)):
        prev = stages[i - 1]
        cur = stages[i]
        field_deltas = {
            field: _round6(float(getattr(cur.stage, field)) - float(getattr(prev.stage, field)))
            for field in CLIMATE_FIELDS
        }
        score_deltas = {
            "climate": _round6(cur.climate_holonomy_proxy - prev.climate_holonomy_proxy),
            "polymer": _round6(cur.polymer_persistence_proxy - prev.polymer_persistence_proxy),
            "compartment": _round6(cur.compartment_co_localisation_proxy - prev.compartment_co_localisation_proxy),
            "coupling": _round6(cur.coupling_proxy - prev.coupling_proxy),
        }
        readiness_delta = _round6(cur.protocell_readiness_proxy - prev.protocell_readiness_proxy)
        velocity = readiness_delta
        acceleration = _round6(velocity - prev_velocity)
        prev_velocity = velocity
        changes.append(
            IntervalChange(
                from_stage=prev.stage.name,
                to_stage=cur.stage.name,
                interval_index=i - 1,
                field_deltas=field_deltas,
                climate_delta=score_deltas["climate"],
                polymer_delta=score_deltas["polymer"],
                compartment_delta=score_deltas["compartment"],
                coupling_delta=score_deltas["coupling"],
                readiness_delta=readiness_delta,
                readiness_velocity=velocity,
                readiness_acceleration=acceleration,
                dominant_driver=_dominant_driver(score_deltas),
                change_kind=classify_change_kind(readiness_delta, acceleration),
            )
        )
    return changes


def classify_option(report: DNAToProtocellReport, changes: Sequence[IntervalChange]) -> Dict[str, str]:
    if not report.stages:
        return {"regime": "empty", "tempo": "none", "transition": "none"}
    peak = report.metrics.get("peak_protocell_readiness_proxy", 0.0)
    gain = report.metrics.get("readiness_gain_proxy", 0.0)
    transition = "crosses_threshold" if report.first_transition_stage else "no_crossing"
    if gain >= 0.42:
        tempo = "strong_temporal_gain"
    elif gain >= 0.30:
        tempo = "moderate_temporal_gain"
    elif gain >= 0.15:
        tempo = "weak_temporal_gain"
    else:
        tempo = "stalled_or_regressive"
    final = report.stages[-1].stage
    if final.compartment_pressure >= 0.90 and final.holonomic_recurrence >= 0.90:
        regime = "compartment_recurrence_locked"
    elif final.amphiphile_availability >= 0.80:
        regime = "amphiphile_compartment_dominated"
    elif final.mineral_template >= 0.70:
        regime = "mineral_template_dominated"
    else:
        regime = "mixed_climate_holonomy"
    return {
        "regime": regime,
        "tempo": tempo,
        "transition": transition,
        "peak_band": "near_threshold" if 0.65 <= peak < 0.75 else ("above_threshold" if peak >= 0.75 else "below_threshold"),
    }


def default_temporal_options() -> List[TemporalForcingOption]:
    """Return deterministic environmental options for v0.7."""

    return [
        TemporalForcingOption(
            name="baseline_v06",
            description="Use the unmodified v0.6 relative climate-holonomic trajectory.",
            field_offsets={},
            field_slopes={},
            field_pulses={},
        ),
        TemporalForcingOption(
            name="wet_dry_pulse_amplified",
            description="Amplify hydration cycling and concentration pulses, modelling stronger repeated wet/dry forcing.",
            field_offsets={},
            field_slopes={"holonomic_recurrence": 0.04, "compartment_pressure": 0.03},
            field_pulses={"wet_dry_cycling": 0.10, "mineral_template": 0.05, "thermal_gradient": 0.04},
        ),
        TemporalForcingOption(
            name="mineral_to_amphiphile_handoff",
            description="Shift dominance from mineral templating to amphiphile-supported compartment co-localisation over time.",
            field_offsets={},
            field_slopes={
                "mineral_template": -0.10,
                "amphiphile_availability": 0.10,
                "compartment_pressure": 0.08,
                "holonomic_recurrence": 0.05,
                "uv_stress": -0.08,
            },
            field_pulses={"wet_dry_cycling": 0.04},
        ),
        TemporalForcingOption(
            name="ionic_moderation_window",
            description="Improve ionic moderation and UV shielding while preserving enough cycling for recurrence.",
            field_offsets={},
            field_slopes={
                "ionic_moderation": 0.12,
                "uv_stress": -0.12,
                "holonomic_recurrence": 0.06,
                "hydration": 0.04,
            },
            field_pulses={"wet_dry_cycling": 0.03},
        ),
        TemporalForcingOption(
            name="uv_and_dilution_stress",
            description="Stress-test scenario with excessive UV and dilution pressure, used as a negative control.",
            field_offsets={"uv_stress": 0.10, "hydration": 0.05},
            field_slopes={
                "wet_dry_cycling": -0.12,
                "mineral_template": -0.08,
                "amphiphile_availability": -0.08,
                "compartment_pressure": -0.08,
                "holonomic_recurrence": -0.06,
            },
            field_pulses={},
        ),
    ]


def evaluate_option(option: TemporalForcingOption) -> TemporalOptionReport:
    base = default_climate_holonomic_stages()
    stages = apply_temporal_option(base, option)
    report = compute_dna_to_protocell_transition(stages=stages)
    changes = compute_interval_changes(report)
    velocities = [c.readiness_velocity for c in changes]
    accelerations = [c.readiness_acceleration for c in changes]
    metrics = {
        "peak_readiness": _round6(report.metrics.get("peak_protocell_readiness_proxy", 0.0)),
        "initial_readiness": _round6(report.metrics.get("initial_readiness_proxy", 0.0)),
        "readiness_gain": _round6(report.metrics.get("readiness_gain_proxy", 0.0)),
        "max_velocity": _round6(max(velocities) if velocities else 0.0),
        "min_velocity": _round6(min(velocities) if velocities else 0.0),
        "mean_velocity": _round6(sum(velocities) / len(velocities) if velocities else 0.0),
        "max_acceleration": _round6(max(accelerations) if accelerations else 0.0),
        "transition_detected": report.metrics.get("transition_detected", 0.0),
        "transition_index": report.metrics.get("transition_index", -1.0),
        "threshold_proxy": report.metrics.get("threshold_proxy", 0.0),
    }
    return TemporalOptionReport(
        option=option,
        transition_report=report,
        interval_changes=changes,
        metrics=metrics,
        classification=classify_option(report, changes),
    )


def compute_temporal_climate_options(
    options: Optional[Sequence[TemporalForcingOption]] = None,
    base_current_state_id: str = "NOEMA-20260517T041500-v06-dna-to-protocell-climate-holonomy",
) -> TemporalClimateOptionsReport:
    selected = list(options) if options is not None else default_temporal_options()
    option_reports = [evaluate_option(o) for o in selected]
    best = max(option_reports, key=lambda r: r.metrics["peak_readiness"])
    fastest = max(option_reports, key=lambda r: r.metrics["max_velocity"])
    crossing = [r for r in option_reports if r.transition_report.first_transition_index is not None]
    earliest = min(crossing, key=lambda r: r.transition_report.first_transition_index).option.name if crossing else None
    baseline = compute_dna_to_protocell_transition()
    interpretation = (
        "The v0.7 model says that the DNA-four-block grammar becomes protocell-relevant not by a single static score, "
        "but by its rate of coupling to a changing climate-holonomic environment. The fastest positive interval is the "
        "surface concentration to polymer/vesicle contact regime, while the strongest final readiness comes from handoff "
        "toward amphiphile compartmentalisation, ionic moderation and holonomic recurrence. UV/dilution stress acts as a "
        "negative-control option that reduces tempo and may prevent stable crossing."
    )
    return TemporalClimateOptionsReport(
        schema=SCHEMA,
        base_current_state_id=base_current_state_id,
        baseline_source_schema=baseline.schema,
        core_v06_metrics=dict(baseline.metrics),
        option_reports=option_reports,
        best_option=best.option.name,
        fastest_growth_option=fastest.option.name,
        earliest_transition_option=earliest,
        interpretation=interpretation,
        certification_boundary={
            "semantic_relation_model_claimed": True,
            "time_varying_climate_holonomy_claimed": True,
            "scenario_options_claimed": True,
            "rate_of_change_proxy_claimed": True,
            "empirical_geological_timeline_claimed": False,
            "empirical_abiogenesis_proof_claimed": False,
            "dna_first_historical_claimed": False,
            "thermodynamic_membrane_simulation_claimed": False,
            "quantum_chemistry_claimed": False,
        },
    )


def run_temporal_climate_options_demo(as_noema: bool = False) -> Dict[str, Any] | TemporalClimateOptionsReport:
    report = compute_temporal_climate_options()
    return report.to_noema_candidate() if as_noema else report
