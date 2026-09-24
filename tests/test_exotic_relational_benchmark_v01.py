from __future__ import annotations

from copy import deepcopy

import pytest

from origins.exobiology.benchmark import (
    ABLATIONS,
    BENCHMARK_SCHEMA,
    COMPARISON_SCOPE,
    benchmark_manifest,
    run_exotic_relational_case,
    run_matched_exotic_relational_benchmark,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_manifest_forbids_ranking_and_raw_cross_chemistry_claims() -> None:
    manifest = benchmark_manifest()
    assert manifest["schema"] == BENCHMARK_SCHEMA
    assert manifest["comparison_scope"] == COMPARISON_SCOPE
    assert manifest["ranking_allowed"] is False
    assert manifest["matched_seed_required"] is True
    assert list(manifest["ablations"]) == list(ABLATIONS)


def test_matched_benchmark_covers_two_runtimes_five_ablations_two_seeds() -> None:
    results = run_matched_exotic_relational_benchmark(
        [deepcopy(SCENARIO_C), deepcopy(SCENARIO_D)],
        seeds=(3, 7),
        steps=40,
        Nx=12,
        Ny=12,
    )
    assert len(results) == 2 * len(ABLATIONS) * 2
    assert {r.profile for r in results} == {
        "AMMONIA_CANDIDATE",
        "HYDROCARBON_CANDIDATE",
    }
    assert all(r.ranking_allowed is False for r in results)
    assert all(r.physical_binding == "OPEN" for r in results)


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_material_conservation_residual_remains_small_across_all_ablations(scenario) -> None:
    for ablation in ABLATIONS:
        result = run_exotic_relational_case(
            deepcopy(scenario),
            ablation=ablation,
            seed=17,
            steps=160,
            Nx=16,
            Ny=16,
        )
        tolerance = 2e-9 * max(1.0, abs(result.material_initial))
        assert result.material_residual_abs <= tolerance


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_no_external_energy_input_has_zero_positive_throughput(scenario) -> None:
    result = run_exotic_relational_case(
        deepcopy(scenario),
        ablation="NO_EXTERNAL_ENERGY_INPUT",
        seed=19,
        steps=80,
        Nx=14,
        Ny=14,
    )
    assert result.energy_throughput_integral == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_no_boundary_assembly_produces_no_boundary_mass_or_compartments(scenario) -> None:
    result = run_exotic_relational_case(
        deepcopy(scenario),
        ablation="NO_BOUNDARY_ASSEMBLY",
        seed=29,
        steps=100,
        Nx=14,
        Ny=14,
    )
    assert result.boundary_mass == pytest.approx(0.0, abs=1e-15)
    assert result.candidate_compartment_count == 0
    assert result.candidate_compartment_area_pixels == 0


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_no_inheritance_keeps_trait_state_zero_from_zero_initial_condition(scenario) -> None:
    result = run_exotic_relational_case(
        deepcopy(scenario),
        ablation="NO_INHERITANCE",
        seed=31,
        steps=120,
        Nx=14,
        Ny=14,
    )
    assert result.trait_mean == pytest.approx(0.0, abs=1e-15)
    assert result.trait_variance == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_benchmark_is_deterministic_for_same_seed(scenario) -> None:
    a = run_exotic_relational_case(
        deepcopy(scenario),
        ablation="BASELINE",
        seed=41,
        steps=90,
        Nx=12,
        Ny=12,
    )
    b = run_exotic_relational_case(
        deepcopy(scenario),
        ablation="BASELINE",
        seed=41,
        steps=90,
        Nx=12,
        Ny=12,
    )
    assert a == b


def test_unknown_ablation_fails_closed() -> None:
    with pytest.raises(ValueError):
        run_exotic_relational_case(
            deepcopy(SCENARIO_C),
            ablation="MAGIC",
            seed=1,
            steps=1,
            Nx=8,
            Ny=8,
        )
