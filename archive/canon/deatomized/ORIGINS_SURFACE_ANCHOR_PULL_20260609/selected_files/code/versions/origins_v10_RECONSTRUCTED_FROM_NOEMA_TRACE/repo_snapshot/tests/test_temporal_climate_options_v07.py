from origins.abiogenesis.temporal_climate_options import (
    compute_temporal_climate_options,
    default_temporal_options,
    run_temporal_climate_options_demo,
)


def test_temporal_options_report_contains_scenarios():
    report = compute_temporal_climate_options()
    assert report.schema == "ORIGINS_NOEMA_TEMPORAL_CLIMATE_OPTIONS_V0_7"
    names = [r.option.name for r in report.option_reports]
    assert "baseline_v06" in names
    assert "mineral_to_amphiphile_handoff" in names
    assert len(names) >= 5


def test_temporal_rates_are_computed_from_v06_stages():
    report = compute_temporal_climate_options()
    baseline = next(r for r in report.option_reports if r.option.name == "baseline_v06")
    assert len(baseline.interval_changes) == 4
    assert baseline.metrics["max_velocity"] > 0.10
    assert baseline.interval_changes[0].change_kind in {"accelerating_gain", "fast_gain_decelerating"}


def test_stress_option_is_not_best():
    report = compute_temporal_climate_options()
    stress = next(r for r in report.option_reports if r.option.name == "uv_and_dilution_stress")
    best = next(r for r in report.option_reports if r.option.name == report.best_option)
    assert best.metrics["peak_readiness"] >= stress.metrics["peak_readiness"]
    assert report.fastest_growth_option in [r.option.name for r in report.option_reports]


def test_temporal_noema_candidate_boundary():
    candidate = run_temporal_climate_options_demo(as_noema=True)
    assert candidate["schema"] == "NOEMA_INGEST_CANDIDATE_V0_1"
    payload = candidate["payload"]
    assert payload["certification_boundary"]["rate_of_change_proxy_claimed"] is True
    assert payload["certification_boundary"]["empirical_geological_timeline_claimed"] is False
    assert payload["certification_boundary"]["empirical_abiogenesis_proof_claimed"] is False


def test_temporal_option_names_are_stable():
    assert [o.name for o in default_temporal_options()] == [
        "baseline_v06",
        "wet_dry_pulse_amplified",
        "mineral_to_amphiphile_handoff",
        "ionic_moderation_window",
        "uv_and_dilution_stress",
    ]
