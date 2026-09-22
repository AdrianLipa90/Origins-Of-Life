"""Matched first-RNA candidate ablation.

This analysis asks whether the current oligomer chemistry reaches an empirical
polymerase-size reference (45 nt) with the synthetic geometry coupling disabled
versus the explicitly retained legacy candidate coupling.

It does not infer catalytic function or self-replication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..biology.first_rna import (
    L_QT45_REFERENCE,
    simulate_first_rna,
)


MODES = ("chemistry_only", "legacy_geometry_candidate")


def _max_expected_length(counts: np.ndarray, threshold: float = 1.0) -> int:
    idx = np.flatnonzero(np.asarray(counts, dtype=float) >= float(threshold))
    return int(idx[-1] + 1) if idx.size else 0


def run_first_rna_candidate_ablation(
    *,
    seeds: list[int],
    temp_C: float = 65.0,
    k_catalysis: float = 7.5,
    pH: float = 7.5,
    concentration_boost: float = 1000.0,
    drying_cycle_h: float = 12.0,
    drying_fraction: float = 0.3,
    hours: float = 500.0,
    dt_h: float = 0.5,
    topo_strength: float = 0.25,
    topo_pulsing: bool = True,
) -> pd.DataFrame:
    if not seeds:
        raise ValueError("at least one seed is required")

    rows: list[dict[str, object]] = []
    for seed in [int(x) for x in seeds]:
        for mode in MODES:
            geometry_mode = (
                "off" if mode == "chemistry_only" else "legacy_candidate"
            )
            state = simulate_first_rna(
                temp_C=temp_C,
                k_catalysis=k_catalysis,
                pH=pH,
                concentration_boost=concentration_boost,
                drying_cycle_h=drying_cycle_h,
                drying_fraction=drying_fraction,
                hours=hours,
                dt_h=dt_h,
                topo_strength=topo_strength,
                topo_pulsing=topo_pulsing,
                seed=seed,
                verbose=False,
                replicator_mode="candidate_only",
                geometry_mode=geometry_mode,
            )
            pool = state.oligomer_pool
            rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "geometry_mode": geometry_mode,
                    "candidate_reference_nt": L_QT45_REFERENCE,
                    "candidate_reached": (
                        state.first_polymerase_size_candidate_t is not None
                    ),
                    "first_candidate_h": state.first_polymerase_size_candidate_t,
                    "final_reference_candidates": (
                        state.n_polymerase_size_candidates
                    ),
                    "final_mean_length_nt": pool.mean_length(),
                    "final_max_length_ge_1_expected": _max_expected_length(
                        pool.counts, threshold=1.0
                    ),
                    "final_n_ge_35": pool.n_above_threshold(35),
                    "final_n_ge_45": pool.n_above_threshold(45),
                    "final_n_ge_50": pool.n_above_threshold(50),
                    "final_total_nucleotide_units": pool.total_monomer_units(),
                    "functional_replication_status": (
                        state.functional_replication_status
                    ),
                    "functional_replicator_t": state.first_replicator_t,
                    "final_functional_replicators": state.n_replicators,
                }
            )

    return pd.DataFrame(rows)
