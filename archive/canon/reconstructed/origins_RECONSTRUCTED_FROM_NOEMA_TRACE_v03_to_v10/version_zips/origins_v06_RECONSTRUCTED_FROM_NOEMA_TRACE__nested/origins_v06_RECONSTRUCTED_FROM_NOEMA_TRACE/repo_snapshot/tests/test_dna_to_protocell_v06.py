from origins.abiogenesis.dna_to_protocell import (
    compute_dna_to_protocell_transition,
    default_climate_holonomic_stages,
    run_dna_to_protocell_demo,
)


def test_dna_to_protocell_default_transition_crosses_only_late():
    report = compute_dna_to_protocell_transition()
    assert report.schema == "ORIGINS_NOEMA_DNA_TO_PROTOCELL_V0_6"
    assert report.metrics["stage_count"] == 5.0
    assert report.first_transition_stage == "protocell_candidate"
    assert report.first_transition_index == 4
    assert report.metrics["transition_detected"] == 1.0
    assert report.stages[-1].crosses_threshold is True
    assert all(not stage.crosses_threshold for stage in report.stages[:-1])


def test_dna_to_protocell_readiness_increases_over_time():
    report = compute_dna_to_protocell_transition()
    readiness = [stage.protocell_readiness_proxy for stage in report.stages]
    assert readiness[-1] > readiness[0]
    assert report.metrics["readiness_gain_proxy"] > 0.35


def test_dna_to_protocell_noema_candidate_boundary():
    candidate = run_dna_to_protocell_demo(as_noema=True)
    assert candidate["schema"] == "NOEMA_INGEST_CANDIDATE_V0_1"
    payload = candidate["payload"]
    assert payload["certification_boundary"]["protocell_readiness_proxy_claimed"] is True
    assert payload["certification_boundary"]["empirical_abiogenesis_proof_claimed"] is False
    assert payload["certification_boundary"]["dna_first_historical_claimed"] is False


def test_default_climate_stage_order_is_stable():
    stages = default_climate_holonomic_stages()
    assert [s.relative_time_order for s in stages] == list(range(len(stages)))
    assert stages[0].name == "alphabet_only_open_surface"
    assert stages[-1].name == "protocell_candidate"
