import numpy as np

from origins.biology.first_rna import (
    EmergenceState,
    L_QT45_REFERENCE,
    OligomerPool,
    simulate_first_rna,
    step_replication,
)


def _state_with_reference_candidates(count: float = 10.0) -> EmergenceState:
    pool = OligomerPool.seed(monomer_conc=1000.0, max_len=80)
    pool.counts[:] = 0.0
    pool.counts[L_QT45_REFERENCE - 1] = count
    pool.monomer_pool = 1000.0 - count * L_QT45_REFERENCE
    state = EmergenceState(oligomer_pool=pool)
    state.gc_mean = 0.50
    return state


def test_candidate_only_does_not_infer_function_from_length_and_gc():
    state = _state_with_reference_candidates(10.0)
    state.t_h = 12.5

    step_replication(
        state,
        temp_C=65.0,
        dt=1.0,
        rng=np.random.default_rng(1),
        activation_mode="candidate_only",
    )

    assert state.n_polymerase_size_candidates == 10.0
    assert state.first_polymerase_size_candidate_t == 12.5
    assert state.first_replicator_t is None
    assert state.n_replicators == 0.0
    assert state.functional_replication_status == "UNRESOLVED_SEQUENCE_AND_ACTIVITY"


def test_legacy_phenomenological_activation_is_explicit():
    state = _state_with_reference_candidates(10.0)
    state.t_h = 4.0

    step_replication(
        state,
        temp_C=65.0,
        dt=1.0,
        rng=np.random.default_rng(1),
        activation_mode="legacy_phenomenological",
    )

    assert state.first_polymerase_size_candidate_t == 4.0
    assert state.first_replicator_t == 4.0
    assert state.n_replicators > 0.0
    assert state.functional_replication_status == "LEGACY_PHENOMENOLOGICAL_MODEL"


def test_default_first_rna_mode_is_fail_closed_on_function():
    state = simulate_first_rna(
        hours=1.0,
        dt_h=0.5,
        seed=7,
        verbose=False,
        topo_strength=0.0,
        topo_pulsing=False,
    )

    assert state.replicator_activation_mode == "candidate_only"
    assert state.functional_replication_status == "UNRESOLVED_SEQUENCE_AND_ACTIVITY"
    assert state.first_replicator_t is None
    assert state.n_replicators == 0.0


def test_length_one_seed_partition_preserves_declared_monomer_units():
    pool = OligomerPool.seed(monomer_conc=1000.0, max_len=80)
    assert pool.counts[0] == 100.0
    assert pool.monomer_pool == 900.0
    assert pool.total_monomer_units() == 1000.0
