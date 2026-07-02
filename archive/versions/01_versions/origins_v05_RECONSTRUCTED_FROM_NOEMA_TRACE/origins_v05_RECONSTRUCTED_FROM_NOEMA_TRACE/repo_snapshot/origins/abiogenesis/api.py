"""Canonical high-level API for repo-facing use.

Prefer this module over direct public use of `origins.orbital` names.
"""

from ..analysis.sweep_orbital import (
    export_orbital_analysis_snapshot,
    run_all_scenarios_orbital,
    run_sweep_v3_orbital,
    run_topo_sweep_orbital,
)
from ..simulator.factory import available_simulator_modes, create_simulator
from ..simulator.universal_orbital import OrbitalUniversalOriginSimulator
from . import AbiogenesisRuntimeAdapter, AbiogenesisRunBundle


def create_abiogenesis_runtime(config, mode: str = "orbital", **kwargs):
    return create_simulator(config=config, mode=mode, **kwargs)


def run_habitat_scan(**kwargs):
    return run_sweep_v3_orbital(**kwargs)


def run_feasibility_scan(**kwargs):
    return run_topo_sweep_orbital(**kwargs)


def run_origin_comparison(**kwargs):
    return run_all_scenarios_orbital(**kwargs)


def export_historical_state(**kwargs):
    return export_orbital_analysis_snapshot(**kwargs)


__all__ = [
    "AbiogenesisRunBundle",
    "AbiogenesisRuntimeAdapter",
    "OrbitalUniversalOriginSimulator",
    "available_simulator_modes",
    "create_abiogenesis_runtime",
    "run_habitat_scan",
    "run_feasibility_scan",
    "run_origin_comparison",
    "export_historical_state",
    "weave_meaning_with_fold",
    "knot_trace_for_noema",
    "run_semantic_nucleic_braiding_demo",
]
from .geometric_semantic_braiding import (
    demo_trace as _semantic_nucleic_demo_trace,
    knot_trace_for_noema,
    weave_meaning_with_fold,
)


def run_semantic_nucleic_braiding_demo(as_noema: bool = False):
    """Run deterministic fold/weave/knot demo for semantic compression tests."""
    trace = _semantic_nucleic_demo_trace()
    return trace.to_noema_candidate() if as_noema else trace


from .semantic_compression import (
    benchmark_semantic_compression,
    compress_braiding_trace,
    detect_semantic_transition,
    run_semantic_compression_demo,
    run_semantic_transition_demo,
)

for _name in [
    "benchmark_semantic_compression",
    "compress_braiding_trace",
    "detect_semantic_transition",
    "run_semantic_compression_demo",
    "run_semantic_transition_demo",
]:
    if _name not in __all__:
        __all__.append(_name)


from .dna_four_blocks import (
    compute_four_block_relation,
    relation_between,
    run_dna_four_block_demo,
)

for _name in [
    "compute_four_block_relation",
    "relation_between",
    "run_dna_four_block_demo",
]:
    if _name not in __all__:
        __all__.append(_name)


from .dna_to_protocell import (
    compute_dna_to_protocell_transition,
    default_climate_holonomic_stages,
    run_dna_to_protocell_demo,
)
for _name in [
    "compute_dna_to_protocell_transition",
    "default_climate_holonomic_stages",
    "run_dna_to_protocell_demo",
]:
    if _name not in __all__:
        __all__.append(_name)

from .temporal_climate_options import (
    compute_temporal_climate_options,
    default_temporal_options,
    run_temporal_climate_options_demo,
)
for _name in [
    "compute_temporal_climate_options",
    "default_temporal_options",
    "run_temporal_climate_options_demo",
]:
    if _name not in __all__:
        __all__.append(_name)
