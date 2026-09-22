import numpy as np

from origins.analysis.first_rna_candidate_v02 import (
    CANDIDATE_REFERENCE_NT,
    CandidateRunConfig,
    _max_length_with_expected_count,
    run_candidate_gate,
    simulate_candidate_length_only,
)
from origins.biology.first_rna import OligomerPool


def test_polymer_free_candidate_initialization_contains_no_chain_above_one_nt():
    pool = OligomerPool.seed(monomer_conc=800.0)
    assert pool.n_above_threshold(2) == 0.0
    assert pool.total_monomer_units() == 800.0


def test_max_length_expected_count_helper():
    pool = OligomerPool.seed(monomer_conc=100.0, max_len=10)
    pool.counts[:] = 0.0
    pool.counts[0] = 5.0
    pool.counts[6] = 1.0
    pool.counts[7] = 0.999
    assert _max_length_with_expected_count(pool) == 7


def test_candidate_only_short_run_never_creates_functional_replicator():
    cfg = CandidateRunConfig(hours=2.0, dt_h=0.5)
    row = simulate_candidate_length_only(seed=301, config=cfg)
    assert row["functional_replication_status"] == "UNRESOLVED_SEQUENCE_AND_ACTIVITY"
    assert row["final_functional_replicators"] == 0.0
    assert row["functional_replicator_t"] is None
    assert np.isfinite(row["final_total_nucleotide_units"])


def test_gate_keeps_45nt_as_length_reference_not_function_claim():
    cfg = CandidateRunConfig(hours=2.0, dt_h=0.5)
    runs, summary = run_candidate_gate(seeds=[301, 302], config=cfg)
    assert CANDIDATE_REFERENCE_NT == 45
    assert summary["physical_function_claim"] is False
    assert summary["geometry_used"] is False
    assert summary["zeta_used"] is False
    assert (runs["final_functional_replicators"] == 0.0).all()
