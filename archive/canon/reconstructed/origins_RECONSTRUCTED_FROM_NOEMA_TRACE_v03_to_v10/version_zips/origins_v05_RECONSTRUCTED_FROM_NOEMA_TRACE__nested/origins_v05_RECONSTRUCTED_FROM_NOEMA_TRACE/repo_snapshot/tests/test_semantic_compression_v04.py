from origins.abiogenesis.api import run_semantic_compression_demo, run_semantic_transition_demo
from origins.abiogenesis.semantic_compression import (
    benchmark_semantic_compression,
    compress_braiding_trace,
    detect_semantic_transition,
)
from origins.abiogenesis.geometric_semantic_braiding import demo_trace


def test_compression_report_keeps_invariant_boundary():
    report = compress_braiding_trace(demo_trace())
    data = report.to_dict()
    assert data["schema"] == "ORIGINS_NOEMA_SEMANTIC_COMPRESSION_V0_4"
    assert data["raw_surface_bytes"] > data["invariant_trace_bytes"]
    assert 0.0 <= data["byte_reduction_proxy"] <= 1.0
    assert 0.0 <= data["semantic_retention_proxy"] <= 1.0
    assert data["certification_boundary"]["byte_exact_compression_claimed"] is False
    assert data["certification_boundary"]["empirical_abiogenesis_proof_claimed"] is False


def test_transition_detector_finds_threshold_crossing():
    reports = benchmark_semantic_compression()
    detection = detect_semantic_transition(reports, threshold=0.50)
    assert detection.detected is True
    assert detection.first_index is not None
    assert detection.transition_score is not None
    assert detection.transition_score >= 0.50


def test_api_exposes_noema_candidates():
    compression = run_semantic_compression_demo(as_noema=True)
    transition = run_semantic_transition_demo(as_noema=True)
    assert compression["schema"] == "NOEMA_INGEST_CANDIDATE_V0_1"
    assert compression["report"]["schema"] == "ORIGINS_NOEMA_SEMANTIC_COMPRESSION_V0_4"
    assert transition["schema"] == "NOEMA_INGEST_CANDIDATE_V0_1"
    assert transition["transition"]["schema"] == "ORIGINS_SEMANTIC_TRANSITION_DETECTION_V0_4"
