from .habitat import OriginHabitatShell, HabitatEmbedding
from .emergence_state import EmergenceCoordinate, EmergenceSystemState
from .feasibility import FeasibilityTerms, compute_feasibility_terms
from .local_clock import compute_emergence_clock
from .recurrence import RecurrenceComponents, compute_recurrence_components
from .residue_memory import HistoricalResidue, HistoricalMemory, apply_residue_update
from .transition import EmergenceTrace, run_emergence_transition
from .repository_semantics import assign_emergence_state_to_entity, build_origin_repository_state
from .execution import AbiogenesisRunBundle, AbiogenesisRuntimeAdapter

__all__ = [
    "OriginHabitatShell",
    "HabitatEmbedding",
    "EmergenceCoordinate",
    "EmergenceSystemState",
    "FeasibilityTerms",
    "compute_feasibility_terms",
    "compute_emergence_clock",
    "RecurrenceComponents",
    "compute_recurrence_components",
    "HistoricalResidue",
    "HistoricalMemory",
    "apply_residue_update",
    "EmergenceTrace",
    "run_emergence_transition",
    "assign_emergence_state_to_entity",
    "build_origin_repository_state",
    "AbiogenesisRunBundle",
    "AbiogenesisRuntimeAdapter",
    "CarbonOrganicFrame",
    "OrganicFrame",
    "MeaningNode",
    "MeaningUnit",
    "MeaningEdge",
    "MeaningRelation",
    "NucleicStrand",
    "NucleicAcidState",
    "FoldState",
    "BraidingMetrics",
    "BraidingTrace",
    "fold_nucleic_geometry",
    "weave_meaning_with_fold",
    "braid_semantics_with_nucleic_acid",
    "knot_trace_for_noema",
    "semantic_nucleic_demo_trace",
]
from .geometric_semantic_braiding import (
    CarbonOrganicFrame, OrganicFrame, MeaningNode, MeaningUnit, MeaningEdge, MeaningRelation,
    NucleicStrand, NucleicAcidState, FoldState, BraidingMetrics, BraidingTrace,
    fold_nucleic_geometry, weave_meaning_with_fold, braid_semantics_with_nucleic_acid,
    knot_trace_for_noema, demo_trace as semantic_nucleic_demo_trace,
)


from .semantic_compression import (
    InvariantTrace,
    SemanticCompressionReport,
    TransitionDetection,
    benchmark_semantic_compression,
    compress_braiding_trace,
    detect_semantic_transition,
    extract_invariant_trace,
    run_semantic_compression_demo,
    run_semantic_transition_demo,
)
for _name in [
    "InvariantTrace",
    "SemanticCompressionReport",
    "TransitionDetection",
    "benchmark_semantic_compression",
    "compress_braiding_trace",
    "detect_semantic_transition",
    "extract_invariant_trace",
    "run_semantic_compression_demo",
    "run_semantic_transition_demo",
]:
    if _name not in __all__:
        __all__.append(_name)


from .dna_to_protocell import (
    ClimateHolonomicStage,
    ProtocellStageScore,
    DNAToProtocellReport,
    climate_holonomy_score,
    compute_dna_to_protocell_transition,
    compute_information_core_proxy,
    compartment_co_localisation_score,
    default_climate_holonomic_stages,
    polymer_persistence_score,
    run_dna_to_protocell_demo,
    score_stage,
)
for _name in [
    "ClimateHolonomicStage",
    "ProtocellStageScore",
    "DNAToProtocellReport",
    "climate_holonomy_score",
    "compute_dna_to_protocell_transition",
    "compute_information_core_proxy",
    "compartment_co_localisation_score",
    "default_climate_holonomic_stages",
    "polymer_persistence_score",
    "run_dna_to_protocell_demo",
    "score_stage",
]:
    if _name not in __all__:
        __all__.append(_name)

from .temporal_climate_options import (
    TemporalForcingOption,
    IntervalChange,
    TemporalOptionReport,
    TemporalClimateOptionsReport,
    apply_temporal_option,
    compute_interval_changes,
    compute_temporal_climate_options,
    default_temporal_options,
    evaluate_option,
    run_temporal_climate_options_demo,
)
for _name in [
    "TemporalForcingOption",
    "IntervalChange",
    "TemporalOptionReport",
    "TemporalClimateOptionsReport",
    "apply_temporal_option",
    "compute_interval_changes",
    "compute_temporal_climate_options",
    "default_temporal_options",
    "evaluate_option",
    "run_temporal_climate_options_demo",
]:
    if _name not in __all__:
        __all__.append(_name)
