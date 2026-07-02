from origins.abiogenesis.geometric_semantic_braiding import (
    CarbonOrganicFrame, MeaningEdge, MeaningNode, NucleicStrand,
    demo_trace, fold_nucleic_geometry, knot_trace_for_noema, weave_meaning_with_fold,
)
from origins.abiogenesis.api import run_semantic_nucleic_braiding_demo


def test_ammonia_solvent_can_still_be_carbon_organic():
    strand = NucleicStrand("AUGC", frame=CarbonOrganicFrame(solvent="ammonia", medium_phase="liquid_ammonia"))
    assert strand.frame.backbone_element == "C"
    assert strand.frame.organic_status == "carbon_based_organic"
    assert strand.frame.solvent == "ammonia"


def test_fold_weave_knot_trace_is_bounded_and_noema_ready():
    trace = demo_trace()
    data = knot_trace_for_noema(trace)
    assert data["schema"] == "ORIGINS_BRAIDING_KNOT_TRACE_V0_3"
    assert 0.0 <= data["metrics"]["identity_retention_proxy"] <= 1.0
    assert 0.0 <= data["metrics"]["knot_stability_proxy"] <= 1.0
    assert data["compression_target"] == "invariant_relation_trace_not_byte_exact_sequence_archive"


def test_api_demo_can_emit_noema_candidate():
    candidate = run_semantic_nucleic_braiding_demo(as_noema=True)
    assert candidate["schema"] == "NOEMA_INGEST_CANDIDATE_V0_1"
    assert candidate["certification_boundary"]["empirical_abiogenesis_proof_claimed"] is False
    assert candidate["trace"]["schema"] == "ORIGINS_NOEMA_SEMANTIC_NUCLEIC_BRAIDING_V0_3"


def test_custom_weave_accepts_meaning_graph():
    trace = weave_meaning_with_fold(
        NucleicStrand("AUGGCCAU"),
        [MeaningNode("fold", 1.0), MeaningNode("memory", 1.2)],
        [MeaningEdge("fold", "memory", 0.8, "binding")],
    )
    assert trace.fold.loop_count >= 0
    assert trace.metrics.semantic_edge_weight == 0.8
