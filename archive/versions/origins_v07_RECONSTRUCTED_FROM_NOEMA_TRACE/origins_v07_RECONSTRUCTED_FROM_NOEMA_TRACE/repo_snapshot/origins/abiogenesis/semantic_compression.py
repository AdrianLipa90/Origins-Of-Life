"""Semantic compression and transition detection for Origins/NOEMA v0.4.

This module adds the next layer after geometric semantic/nucleic braiding:
it compresses a full braiding trace into stable invariant commitments and
detects a transition threshold where recurrence/residue/identity begin to
behave like proto-memory.

Boundary
--------
This is a deterministic proxy model. It does not claim empirical proof of
abiogenesis, thermodynamic RNA folding, or byte-exact compression. It measures
semantic/invariant retention: whether recurrence, residue, closure and
identity-relevant transition structure survive projection to a compact trace.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .geometric_semantic_braiding import (
    BraidingTrace,
    CarbonOrganicFrame,
    MeaningEdge,
    MeaningNode,
    NucleicStrand,
    demo_trace,
    knot_trace_for_noema,
    weave_meaning_with_fold,
)

SCHEMA = "ORIGINS_NOEMA_SEMANTIC_COMPRESSION_V0_4"


def _clamp01(x: float) -> float:
    if not isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _stable_json_size(obj: Mapping[str, Any]) -> int:
    return len(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


@dataclass(frozen=True)
class InvariantTrace:
    """Compact relation-preserving projection of a braiding trace."""

    schema: str
    sequence_length: int
    organic_status: str
    solvent: str
    medium_phase: str
    semantic_symbols: List[str]
    semantic_edge_kinds: List[str]
    fold_closure_density: float
    fold_crossing_count: int
    fold_loop_count: int
    residue_memory_score: float
    recurrence_retention_proxy: float
    identity_retention_proxy: float
    knot_stability_proxy: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticCompressionReport:
    """Report comparing full trace surface with invariant trace."""

    schema: str
    raw_surface_bytes: int
    invariant_trace_bytes: int
    byte_reduction_proxy: float
    semantic_retention_proxy: float
    recurrence_retention_proxy: float
    residue_consistency_proxy: float
    identity_retention_proxy: float
    transition_readiness_proxy: float
    compression_target: str
    invariant_trace: InvariantTrace
    certification_boundary: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["invariant_trace"] = self.invariant_trace.to_dict()
        return data

    def to_noema_candidate(self) -> Dict[str, Any]:
        return {
            "schema": "NOEMA_INGEST_CANDIDATE_V0_1",
            "namespace": "Origins-Of-Life",
            "project_id": "Origins-Of-Life::semantic_nucleic_braiding",
            "layer": "episodic",
            "snap_type": "iteration_state",
            "tags": ["origins", "semantic_compression", "transition_detector", "nucleic_acid", "braiding"],
            "meaning_candidate": {
                "hypothesis": (
                    "A carbon-organic nucleic-acid braiding trace can be projected into a smaller "
                    "NOEMA invariant trace while retaining recurrence, residue and identity proxies."
                ),
                "confidence": 0.80,
                "source": "local NOEMA CURRENT + Origins v0.4 deterministic semantic compression module",
                "context_required": True,
                "canon_allowed": False,
                "needs_repair": False,
                "uncertainty_operator": "chyba",
            },
            "report": self.to_dict(),
        }


@dataclass(frozen=True)
class TransitionDetection:
    """Threshold event for proto-memory / proto-life style transition."""

    schema: str
    threshold: float
    detected: bool
    first_index: Optional[int]
    previous_score: Optional[float]
    transition_score: Optional[float]
    interpretation: str
    reports: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_noema_candidate(self) -> Dict[str, Any]:
        return {
            "schema": "NOEMA_INGEST_CANDIDATE_V0_1",
            "namespace": "Origins-Of-Life",
            "project_id": "Origins-Of-Life::semantic_nucleic_braiding",
            "layer": "episodic",
            "snap_type": "audit",
            "tags": ["origins", "semantic_compression", "transition", "proto_memory"],
            "meaning_candidate": {
                "hypothesis": (
                    "A transition candidate is detected when semantic retention crosses a defined "
                    "threshold after lower-retention precursor states."
                ),
                "confidence": 0.76 if self.detected else 0.55,
                "source": "deterministic v0.4 transition detector over local benchmark traces",
                "context_required": True,
                "canon_allowed": False,
                "needs_repair": False,
                "uncertainty_operator": "chyba",
            },
            "transition": self.to_dict(),
        }


def extract_invariant_trace(trace: BraidingTrace) -> InvariantTrace:
    """Extract the stable NOEMA-facing invariant from a full braiding trace."""
    metrics = trace.metrics
    return InvariantTrace(
        schema="ORIGINS_INVARIANT_TRACE_V0_4",
        sequence_length=trace.strand.length,
        organic_status=trace.strand.frame.organic_status,
        solvent=trace.strand.frame.solvent,
        medium_phase=trace.strand.frame.medium_phase,
        semantic_symbols=[node.symbol for node in trace.meaning_nodes],
        semantic_edge_kinds=sorted({edge.kind for edge in trace.meaning_edges}),
        fold_closure_density=metrics.closure_density,
        fold_crossing_count=metrics.crossing_number_proxy,
        fold_loop_count=metrics.loop_closure_proxy,
        residue_memory_score=metrics.residue_memory_score,
        recurrence_retention_proxy=metrics.recurrence_retention_proxy,
        identity_retention_proxy=metrics.identity_retention_proxy,
        knot_stability_proxy=metrics.knot_stability_proxy,
    )


def compress_braiding_trace(trace: BraidingTrace) -> SemanticCompressionReport:
    """Compress full trace surface to an invariant relation trace."""
    raw = trace.to_dict()
    invariant = extract_invariant_trace(trace)
    invariant_dict = invariant.to_dict()
    raw_bytes = _stable_json_size(raw)
    invariant_bytes = _stable_json_size(invariant_dict)
    byte_reduction = 0.0 if raw_bytes <= 0 else _clamp01(1.0 - invariant_bytes / raw_bytes)
    m = trace.metrics
    semantic_retention = _clamp01(
        0.28 * m.identity_retention_proxy
        + 0.24 * m.recurrence_retention_proxy
        + 0.22 * m.residue_memory_score
        + 0.16 * m.knot_stability_proxy
        + 0.10 * m.closure_density
    )
    transition_readiness = _clamp01(
        0.36 * semantic_retention
        + 0.24 * m.knot_stability_proxy
        + 0.20 * m.recurrence_retention_proxy
        + 0.20 * m.residue_memory_score
    )
    return SemanticCompressionReport(
        schema=SCHEMA,
        raw_surface_bytes=raw_bytes,
        invariant_trace_bytes=invariant_bytes,
        byte_reduction_proxy=byte_reduction,
        semantic_retention_proxy=semantic_retention,
        recurrence_retention_proxy=m.recurrence_retention_proxy,
        residue_consistency_proxy=m.residue_memory_score,
        identity_retention_proxy=m.identity_retention_proxy,
        transition_readiness_proxy=transition_readiness,
        compression_target="invariant_relation_trace_not_byte_exact_sequence_archive",
        invariant_trace=invariant,
        certification_boundary={
            "empirical_abiogenesis_proof_claimed": False,
            "thermodynamic_rna_folding_claimed": False,
            "byte_exact_compression_claimed": False,
            "semantic_invariant_projection_claimed": True,
            "carbon_organic_scope": True,
        },
    )


def benchmark_semantic_compression() -> List[SemanticCompressionReport]:
    """Run deterministic benchmark traces from weak to stronger proto-memory proxies."""
    frame = CarbonOrganicFrame(solvent="ammonia_water_mixture", medium_phase="ammonia_aqueous_eutectic")
    cases = [
        (
            "AUAU",
            [MeaningNode("fluctuation", 0.6, 0.0)],
            [],
        ),
        (
            "AUGCAU",
            [MeaningNode("recurrence", 1.0, 0.0), MeaningNode("residue", 0.9, 0.2)],
            [MeaningEdge("recurrence", "residue", 0.7, "recurrence")],
        ),
        (
            "AUGCGUAUCGCA",
            [
                MeaningNode("recurrence", 1.4, 0.0),
                MeaningNode("residue", 1.2, 0.3),
                MeaningNode("transition", 1.0, 0.6),
                MeaningNode("memory", 1.6, 0.9),
            ],
            [
                MeaningEdge("recurrence", "residue", 1.0, "recurrence"),
                MeaningEdge("residue", "memory", 1.3, "residue"),
                MeaningEdge("transition", "memory", 0.9, "transition"),
                MeaningEdge("memory", "recurrence", 0.7, "retention"),
            ],
        ),
    ]
    reports: List[SemanticCompressionReport] = []
    for seq, nodes, edges in cases:
        trace = weave_meaning_with_fold(NucleicStrand(seq, frame=frame), nodes, edges)
        reports.append(compress_braiding_trace(trace))
    return reports


def detect_semantic_transition(
    reports: Sequence[SemanticCompressionReport], *, threshold: float = 0.50
) -> TransitionDetection:
    """Find first crossing of transition readiness threshold."""
    if not reports:
        raise ValueError("at least one compression report is required")
    previous_score: Optional[float] = None
    for index, report in enumerate(reports):
        score = report.transition_readiness_proxy
        if score >= threshold and (previous_score is None or previous_score < threshold):
            return TransitionDetection(
                schema="ORIGINS_SEMANTIC_TRANSITION_DETECTION_V0_4",
                threshold=threshold,
                detected=True,
                first_index=index,
                previous_score=previous_score,
                transition_score=score,
                interpretation=(
                    "transition candidate: invariant semantic retention crossed threshold; "
                    "this is proto-memory/proto-life proxy, not empirical proof"
                ),
                reports=[r.to_dict() for r in reports],
            )
        previous_score = score
    return TransitionDetection(
        schema="ORIGINS_SEMANTIC_TRANSITION_DETECTION_V0_4",
        threshold=threshold,
        detected=False,
        first_index=None,
        previous_score=previous_score,
        transition_score=None,
        interpretation="no threshold crossing detected in deterministic benchmark",
        reports=[r.to_dict() for r in reports],
    )


def run_semantic_compression_demo(as_noema: bool = False) -> Any:
    """Compress the existing v0.3 demo trace."""
    report = compress_braiding_trace(demo_trace())
    return report.to_noema_candidate() if as_noema else report


def run_semantic_transition_demo(as_noema: bool = False) -> Any:
    """Run benchmark and detect semantic transition."""
    detection = detect_semantic_transition(benchmark_semantic_compression())
    return detection.to_noema_candidate() if as_noema else detection


__all__ = [
    "InvariantTrace",
    "SemanticCompressionReport",
    "TransitionDetection",
    "extract_invariant_trace",
    "compress_braiding_trace",
    "benchmark_semantic_compression",
    "detect_semantic_transition",
    "run_semantic_compression_demo",
    "run_semantic_transition_demo",
]
