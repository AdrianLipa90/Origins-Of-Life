"""Matched 2x2 ablation for experimental geometry and zeta operators.

The design isolates four arms with identical chemistry parameters and seeds:

    chemistry_only : geometry=0, zeta=0
    zeta_only      : geometry=0, zeta=1
    geometry_only  : geometry=1, zeta=0
    full           : geometry=1, zeta=1

This is a software-model ablation. It does not establish physical binding of
CP1/Bloch geometry or zeta-zero indexing to abiogenesis.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..scenarios import ScenarioConfig
from ..simulator import UniversalOriginSimulator


ARM_ORDER = ("chemistry_only", "zeta_only", "geometry_only", "full")
METRICS = (
    "mean_R",
    "mean_M",
    "max_R",
    "max_M",
    "polymer_threshold_pixels",
    "membrane_threshold_pixels",
    "n_polymers",
    "n_protocells",
    "protocell_area_pixels",
    "nucleotide_material_total",
    "berry_accumulated",
    "bloch_coherence",
)


def _arm_config(base: ScenarioConfig, arm: str, seed: int) -> ScenarioConfig:
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown ablation arm: {arm!r}")

    cfg = deepcopy(base)
    cfg.seed = int(seed)
    geometry_enabled = arm in {"geometry_only", "full"}
    zeta_enabled = arm in {"zeta_only", "full"}

    cfg.topo_strength = float(base.topo_strength) if geometry_enabled else 0.0
    cfg.use_zeta_constraints = bool(zeta_enabled)
    return cfg


def _update_digest(h: "hashlib._Hash", value: np.ndarray) -> None:
    arr = np.ascontiguousarray(value)
    h.update(str(arr.dtype).encode("ascii"))
    h.update(repr(arr.shape).encode("ascii"))
    h.update(arr.view(np.uint8).tobytes())


def initial_state_digest(sim: UniversalOriginSimulator) -> str:
    """Digest state that must be identical before experimental operators act."""
    h = hashlib.sha256()
    for name in ("E", "O", "N", "N_surface", "R", "M", "L", "Cat"):
        value = getattr(sim, name)
        if value is None:
            raise RuntimeError(f"{name} is not initialized")
        _update_digest(h, value)

    pop = sim.rna_population
    for value in (pop.pos_x, pop.pos_y, pop.length, pop.fitness):
        _update_digest(h, value)
    h.update(str(pop.size).encode("ascii"))
    return h.hexdigest()


def _final_metrics(sim: UniversalOriginSimulator) -> dict[str, float | int]:
    return {
        "mean_R": float(np.mean(sim.R)),
        "mean_M": float(np.mean(sim.M)),
        "max_R": float(np.max(sim.R)),
        "max_M": float(np.max(sim.M)),
        "polymer_threshold_pixels": int(
            np.count_nonzero(sim.R > sim.protocell_detector.threshold_R)
        ),
        "membrane_threshold_pixels": int(
            np.count_nonzero(sim.M > sim.protocell_detector.threshold_M)
        ),
        "n_polymers": int(sim.rna_population.size),
        "n_protocells": int(sim.protocell_count),
        "protocell_area_pixels": int(sim.protocell_area_pixels),
        "nucleotide_material_total": float(sim.nucleotide_material_total()),
        "berry_accumulated": float(sim.topo.berry_accumulated),
        "bloch_coherence": float(sim.topo.bloch_coherence()),
    }


def factorial_effects(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute per-seed 2x2 contrasts without assigning physical meaning."""
    rows: list[dict[str, float | int | str]] = []
    for seed, group in runs.groupby("seed", sort=True):
        by_arm = group.set_index("arm")
        missing = [arm for arm in ARM_ORDER if arm not in by_arm.index]
        if missing:
            raise ValueError(f"seed {seed}: missing arms {missing}")

        for metric in METRICS:
            c = float(by_arm.loc["chemistry_only", metric])
            z = float(by_arm.loc["zeta_only", metric])
            g = float(by_arm.loc["geometry_only", metric])
            f = float(by_arm.loc["full", metric])
            rows.append(
                {
                    "seed": int(seed),
                    "metric": metric,
                    "zeta_at_zero_geometry": z - c,
                    "geometry_at_zero_zeta": g - c,
                    "interaction": f - g - z + c,
                    "full_minus_chemistry": f - c,
                }
            )
    return pd.DataFrame(rows)


def run_matched_factorial_ablation(
    config: ScenarioConfig,
    *,
    seeds: Sequence[int] | None = None,
    Nx: int = 32,
    Ny: int = 32,
    dt_h: float = 0.05,
    hours: float = 12.0,
    record_interval: float | None = None,
    include_clay: bool = True,
    preseed_rna: bool = True,
    outdir: str = "outputs_ablation",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run matched chemistry/geometry/zeta/full arms.

    Every arm for a given seed must have the same pre-operator chemical and RNA
    initial state. A mismatch fails closed instead of returning an ablation.
    """
    run_seeds = [int(config.seed)] if seeds is None else [int(s) for s in seeds]
    if not run_seeds:
        raise ValueError("at least one seed is required")
    if Nx <= 0 or Ny <= 0 or dt_h <= 0 or hours <= 0:
        raise ValueError("Nx, Ny, dt_h and hours must be positive")

    rec = max(dt_h, hours) if record_interval is None else float(record_interval)
    rows: list[dict[str, float | int | str | bool]] = []

    for seed in run_seeds:
        digest_by_arm: dict[str, str] = {}
        sims: dict[str, UniversalOriginSimulator] = {}

        for arm in ARM_ORDER:
            cfg = _arm_config(config, arm, seed)
            sim = UniversalOriginSimulator(
                cfg,
                Nx=Nx,
                Ny=Ny,
                dt_h=dt_h,
                outdir=f"{outdir}/seed_{seed}/{arm}",
                include_clay=include_clay,
                preseed_rna=preseed_rna,
            )
            sim.initialize()
            digest_by_arm[arm] = initial_state_digest(sim)
            sims[arm] = sim

        if len(set(digest_by_arm.values())) != 1:
            raise RuntimeError(
                f"seed {seed}: unmatched initial states: {digest_by_arm}"
            )

        digest = next(iter(digest_by_arm.values()))
        for arm in ARM_ORDER:
            sim = sims[arm]
            sim.run(hours=hours, record_interval=rec, verbose=False)
            row: dict[str, float | int | str | bool] = {
                "scenario": config.code,
                "seed": seed,
                "arm": arm,
                "geometry_enabled": arm in {"geometry_only", "full"},
                "zeta_enabled": arm in {"zeta_only", "full"},
                "initial_state_digest": digest,
            }
            row.update(_final_metrics(sim))
            rows.append(row)

    runs = pd.DataFrame(rows)
    effects = factorial_effects(runs)
    return runs, effects
