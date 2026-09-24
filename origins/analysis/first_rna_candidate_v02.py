"""Candidate-length-only audit for the legacy first-RNA chemistry.

This module intentionally does not call step_replication and does not construct
TopologyField. It tests the monomer/oligomer chemistry before any claim about
sequence function or replication.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd

from ..biology.first_rna import (
    EmergenceState,
    OligomerPool,
    k_hydrolysis_effective,
    k_ligation_effective,
    step_oligomer_pool,
)


CANDIDATE_REFERENCE_NT = 45
FUNCTIONAL_REPLICATION_STATUS = "UNRESOLVED_SEQUENCE_AND_ACTIVITY"


@dataclass(frozen=True)
class CandidateRunConfig:
    hours: float = 500.0
    dt_h: float = 0.5
    temp_C: float = 65.0
    pH: float = 7.5
    k_catalysis: float = 7.5
    concentration_boost: float = 1000.0
    drying_cycle_h: float = 12.0
    drying_fraction: float = 0.30
    initial_nucleotide_units: float = 800.0


def _max_length_with_expected_count(
    pool: OligomerPool,
    threshold: float = 1.0,
) -> int:
    idx = np.flatnonzero(pool.counts >= float(threshold))
    if idx.size == 0:
        return 0
    return int(idx[-1] + 1)


def _validate_pool(pool: OligomerPool) -> None:
    if not np.isfinite(pool.counts).all():
        raise FloatingPointError("candidate pool contains NaN/Inf")
    if not math.isfinite(float(pool.monomer_pool)):
        raise FloatingPointError("candidate monomer pool contains NaN/Inf")
    if np.any(pool.counts < -1e-10) or pool.monomer_pool < -1e-10:
        raise FloatingPointError("candidate pool contains negative state")


def simulate_candidate_length_only(
    *,
    seed: int,
    config: CandidateRunConfig = CandidateRunConfig(),
) -> dict[str, object]:
    """Run legacy oligomer chemistry with geometry and replication disabled."""
    if config.hours <= 0 or config.dt_h <= 0:
        raise ValueError("hours and dt_h must be positive")
    if not (0.0 <= config.drying_fraction <= 1.0):
        raise ValueError("drying_fraction must lie in [0, 1]")
    if config.drying_cycle_h <= 0:
        raise ValueError("drying_cycle_h must be positive")

    rng = np.random.default_rng(int(seed))
    state = EmergenceState(
        oligomer_pool=OligomerPool.seed(
            monomer_conc=float(config.initial_nucleotide_units)
        )
    )
    state.n_replicators = 0.0
    state.first_replicator_t = None

    initial_total = state.oligomer_pool.total_monomer_units()
    if abs(initial_total - config.initial_nucleotide_units) > 1e-9:
        raise RuntimeError("candidate initialization changed nucleotide units")

    k_hyd = k_hydrolysis_effective(config.temp_C, config.pH)
    n_steps = int(config.hours / config.dt_h)

    for step in range(n_steps):
        state.t_h = step * config.dt_h
        phase = (state.t_h % config.drying_cycle_h) / config.drying_cycle_h
        is_dry = phase < config.drying_fraction

        boost = (
            config.concentration_boost * 10.0
            if is_dry
            else config.concentration_boost
        )
        k_lig = k_ligation_effective(
            config.temp_C,
            config.k_catalysis,
            0.0,  # geometry/Bloch disabled
            0.0,  # Berry disabled
            state.gc_mean,
            concentration_boost=boost,
        )
        k_hyd_eff = k_hyd * 0.02 if is_dry else k_hyd
        step_oligomer_pool(
            state,
            k_lig,
            k_hyd_eff,
            config.dt_h,
            rng,
        )
        _validate_pool(state.oligomer_pool)

        # Replication is intentionally absent from this causal stage.
        if state.n_replicators != 0.0 or state.first_replicator_t is not None:
            raise RuntimeError("candidate-only stage created a functional replicator")

    pool = state.oligomer_pool
    final_total = pool.total_monomer_units()
    return {
        "seed": int(seed),
        "candidate_reference_nt": CANDIDATE_REFERENCE_NT,
        "candidate_reached": bool(
            pool.n_above_threshold(CANDIDATE_REFERENCE_NT) >= 1.0
        ),
        "final_reference_candidates": float(
            pool.n_above_threshold(CANDIDATE_REFERENCE_NT)
        ),
        "final_mean_length_nt": float(pool.mean_length()),
        "final_max_length_ge_1_expected": _max_length_with_expected_count(pool),
        "final_n_ge_10": float(pool.n_above_threshold(10)),
        "final_n_ge_35": float(pool.n_above_threshold(35)),
        "final_n_ge_45": float(pool.n_above_threshold(45)),
        "final_n_ge_50": float(pool.n_above_threshold(50)),
        "initial_total_nucleotide_units": float(initial_total),
        "final_total_nucleotide_units": float(final_total),
        "net_external_nucleotide_input": float(final_total - initial_total),
        "functional_replication_status": FUNCTIONAL_REPLICATION_STATUS,
        "functional_replicator_t": None,
        "final_functional_replicators": 0.0,
    }


def run_candidate_gate(
    *,
    seeds: Sequence[int] = tuple(range(301, 321)),
    config: CandidateRunConfig = CandidateRunConfig(),
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = [
        simulate_candidate_length_only(seed=int(seed), config=config)
        for seed in seeds
    ]
    if not rows:
        raise ValueError("at least one seed is required")

    runs = pd.DataFrame(rows)
    median_max = float(np.median(runs["final_max_length_ge_1_expected"]))
    n_reached = int(runs["candidate_reached"].sum())
    functional_clean = bool(
        (runs["final_functional_replicators"] == 0.0).all()
        and runs["functional_replicator_t"].isna().all()
    )
    finite = bool(
        np.isfinite(
            runs[
                [
                    "final_reference_candidates",
                    "final_mean_length_nt",
                    "final_n_ge_10",
                    "final_n_ge_35",
                    "final_n_ge_45",
                    "final_n_ge_50",
                    "final_total_nucleotide_units",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )

    passed = bool(
        median_max >= CANDIDATE_REFERENCE_NT
        and n_reached >= 10
        and functional_clean
        and finite
    )
    verdict = (
        "PASS_LENGTH_REACHABILITY"
        if passed
        else "FAIL_LENGTH_REACHABILITY"
    )
    summary = {
        "schema": "ORIGINS_FIRST_RNA_CANDIDATE_GATE_V0_2",
        "candidate_reference_nt": CANDIDATE_REFERENCE_NT,
        "seeds": [int(seed) for seed in seeds],
        "median_max_length_ge_1_expected": median_max,
        "seeds_reaching_45nt_expected_count_ge_1": n_reached,
        "functional_replication_status": FUNCTIONAL_REPLICATION_STATUS,
        "functional_replicator_count_is_zero": functional_clean,
        "finite_state_gate": finite,
        "verdict": verdict,
        "physical_function_claim": False,
        "geometry_used": False,
        "zeta_used": False,
    }
    return runs, summary
