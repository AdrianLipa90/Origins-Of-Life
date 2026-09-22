from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.bindings import scenario_config_to_entity_record
from origins.biology.first_rna import (
    EmergenceState,
    OligomerPool,
    _heisenberg_clip,
    step_oligomer_pool,
)
from origins.biology.protocell import ProtocellDetector
from origins.orbital.bundle_builder import build_orbital_bundle_from_simulator
from origins.scenarios import SCENARIO_A
from origins.simulator.universal import UniversalOriginSimulator
from origins.topology.constraints import ZetaRiemannModulator


def test_first_rna_hydrolysis_conserves_nucleotide_units_plus_declared_input() -> None:
    pool = OligomerPool.seed(monomer_conc=1000.0, max_len=20)
    pool.counts[:] = 0.0
    pool.counts[7] = 12.0  # twelve 8-mers
    pool.monomer_pool = 100.0
    state = EmergenceState(oligomer_pool=pool)
    before = pool.total_monomer_units()

    step_oligomer_pool(
        state,
        k_lig=0.0,
        k_hyd=0.25,
        dt=0.1,
        rng=np.random.default_rng(1),
    )

    # The only external material source in this step is 5 monomer-units/hour.
    assert state.oligomer_pool.total_monomer_units() == pytest.approx(
        before + 0.5,
        rel=1e-10,
        abs=1e-10,
    )


def test_first_rna_nonfinite_state_fails_loud() -> None:
    with pytest.raises(FloatingPointError):
        _heisenberg_clip(float("nan"))

    state = EmergenceState(oligomer_pool=OligomerPool.seed(max_len=10))
    state.oligomer_pool.counts[2] = np.nan
    with pytest.raises(FloatingPointError):
        step_oligomer_pool(
            state,
            k_lig=0.0,
            k_hyd=0.0,
            dt=0.1,
            rng=np.random.default_rng(2),
        )


def test_clay_adsorption_moves_material_without_multiplying_it() -> None:
    sim = UniversalOriginSimulator(
        deepcopy(SCENARIO_A),
        Nx=16,
        Ny=16,
        dt_h=0.1,
        include_clay=True,
        preseed_rna=False,
    )
    sim.initialize()
    before = float(np.sum(sim.N) + np.sum(sim.N_surface))
    sim._clay_catalysis_explicit()
    after = float(np.sum(sim.N) + np.sum(sim.N_surface))
    assert after == pytest.approx(before, rel=1e-7, abs=1e-7)
    assert float(np.sum(sim.N_surface)) >= 0.0


def test_polymerization_conserves_field_level_nucleotide_material() -> None:
    sim = UniversalOriginSimulator(
        deepcopy(SCENARIO_A),
        Nx=16,
        Ny=16,
        dt_h=0.05,
        include_clay=True,
        preseed_rna=False,
    )
    sim.initialize()
    sim._clay_catalysis_explicit()
    before = sim.nucleotide_material_total()
    sim.step_polymerization()
    after = sim.nucleotide_material_total()
    assert after == pytest.approx(before, rel=1e-6, abs=1e-6)


def test_polymer_free_initial_condition_is_available_explicitly() -> None:
    sim = UniversalOriginSimulator(
        deepcopy(SCENARIO_A),
        Nx=12,
        Ny=12,
        preseed_rna=False,
        include_clay=False,
    )
    sim.initialize()
    assert sim.rna_population.size == 0
    assert float(np.sum(sim.R)) == pytest.approx(0.0)


def test_protocell_observable_counts_connected_structures_not_pixels() -> None:
    det = ProtocellDetector(threshold_M=0.05, threshold_R=0.03)
    M = np.zeros((12, 12), dtype=float)
    R = np.zeros((12, 12), dtype=float)
    M[1:3, 1:3] = 0.2
    R[1:3, 1:3] = 0.2
    M[8:10, 8:10] = 0.2
    R[8:10, 8:10] = 0.2

    observation = det.detect_components(M, R)
    assert observation["count"] == 2
    assert observation["area_pixels"] == 8

    sim = UniversalOriginSimulator(
        deepcopy(SCENARIO_A),
        Nx=12,
        Ny=12,
        include_clay=False,
        preseed_rna=False,
    )
    sim.initialize()
    sim.M[:] = M
    sim.R[:] = R
    sim.step_protocell_detection()
    assert sim.protocell_count == 2
    assert sim.protocell_area_pixels == 8


def test_zeta_indexed_regularizer_is_explicitly_resolution_independent_in_mapping() -> None:
    mod = ZetaRiemannModulator(sigma_heis=0.0)
    target = mod._normalized_target(14.1347)
    assert 0.0 < target < 0.5

    mask16 = mod.spectral_mask((16, 16))
    mask32 = mod.spectral_mask((32, 32))
    assert mask16.shape == (16, 16)
    assert mask32.shape == (32, 32)
    assert np.isfinite(mask16).all()
    assert np.isfinite(mask32).all()
    assert np.min(mask16) >= 0.0 and np.max(mask16) <= 1.0
    assert np.min(mask32) >= 0.0 and np.max(mask32) <= 1.0


def test_expected_protocell_target_does_not_enter_semantic_mass() -> None:
    cfg = deepcopy(SCENARIO_A)
    cfg.expected_protocells = 999999
    record = scenario_config_to_entity_record(cfg)
    assert record.semantic_mass == pytest.approx(1.0)


def test_orbital_bundle_uses_one_live_coherence_state_and_persists_memory() -> None:
    sim = UniversalOriginSimulator(
        deepcopy(SCENARIO_A),
        Nx=16,
        Ny=16,
        dt_h=0.1,
        include_clay=False,
        preseed_rna=False,
    )
    sim.initialize()
    sim.topo.advance(7.0)

    first = build_orbital_bundle_from_simulator(sim, delta_t=1.0)
    live_coherence = sim.topo.bloch_coherence()
    assert first.coordinate["coherence"] == pytest.approx(live_coherence)
    assert first.coordinate["defect"] == pytest.approx(1.0 - live_coherence)
    assert len(first.memory_state["residues"]) == 1

    second = build_orbital_bundle_from_simulator(sim, delta_t=1.0)
    assert len(second.memory_state["residues"]) == 2


def test_orbital_external_load_is_bounded_coverage_not_raw_count() -> None:
    sim = UniversalOriginSimulator(
        deepcopy(SCENARIO_A),
        Nx=10,
        Ny=10,
        include_clay=False,
        preseed_rna=False,
    )
    sim.initialize()
    sim.protocell_area_pixels = 25
    assert sim.protocell_coverage_fraction() == pytest.approx(0.25)
