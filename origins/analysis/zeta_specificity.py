"""Matched controls for zeta spectral specificity.

This module does not replace the production/legacy ZetaRiemannModulator.
It builds experimental controls that separate:
- no zeta operator,
- zeta RNG noise only,
- narrow zeta-indexed notches,
- narrow reflected-placebo notches,
- the current broad legacy mask.

The reflected placebo preserves target count, mean, span and all pairwise
spacing magnitudes under reflection about the target mean. Any difference
between the two narrow masks therefore depends on absolute spectral placement,
not on generic notch count or spacing geometry.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Sequence

import numpy as np
import pandas as pd

from ..scenarios import ScenarioConfig
from ..simulator import UniversalOriginSimulator
from ..topology.constraints import ZetaRiemannModulator
from .ablation import METRICS, _final_metrics, initial_state_digest


MODES = (
    "none",
    "noise_only",
    "narrow_zeta",
    "narrow_reflected_placebo",
    "legacy_broad",
)


class ControlledZetaModulator(ZetaRiemannModulator):
    """Experimental spectral-control modulator for falsification only."""

    def __init__(
        self,
        *,
        mode: str,
        zeros: list | None = None,
        lambda_soft: float = 5.0,
        sigma_heis: float = 0.001,
        neighbor_cross_talk: float = 0.01,
    ):
        if mode not in MODES[1:]:
            raise ValueError(f"unsupported controlled-zeta mode: {mode!r}")
        if not (0.0 < neighbor_cross_talk < 1.0):
            raise ValueError("neighbor_cross_talk must lie in (0, 1)")
        super().__init__(
            zeros=zeros,
            lambda_soft=lambda_soft,
            sigma_heis=sigma_heis,
        )
        self.mode = mode
        self.neighbor_cross_talk = float(neighbor_cross_talk)

    def mapped_targets(self) -> np.ndarray:
        targets = np.array(
            [self._normalized_target(abs(z.imag)) for z in self.zeros],
            dtype=float,
        )
        if targets.size == 0:
            raise ValueError("at least one spectral target is required")
        if self.mode == "narrow_reflected_placebo":
            mean = float(np.mean(targets))
            targets = 2.0 * mean - targets
        return targets

    def narrow_sigma(self) -> float:
        base = np.sort(
            np.array(
                [self._normalized_target(abs(z.imag)) for z in self.zeros],
                dtype=float,
            )
        )
        if base.size < 2:
            raise ValueError("narrow specificity control requires >=2 targets")
        dmin = float(np.min(np.diff(base)))
        # Choose sigma so a nearest neighboring notch contributes at most the
        # declared cross-talk fraction at the adjacent target center:
        # exp(-0.5*(d/sigma)^2) = cross_talk.
        return dmin / math.sqrt(2.0 * math.log(1.0 / self.neighbor_cross_talk))

    def spectral_mask(self, shape: tuple[int, int]) -> np.ndarray:
        if self.mode == "legacy_broad":
            return super().spectral_mask(shape)

        Nx, Ny = shape
        if Nx <= 0 or Ny <= 0:
            raise ValueError("shape dimensions must be positive")
        if self.mode == "noise_only":
            return np.ones(shape, dtype=float)

        kx = np.fft.fftfreq(Nx)
        ky = np.fft.fftfreq(Ny)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        k_mag = np.sqrt(KX**2 + KY**2)
        sigma = self.narrow_sigma()

        mask = np.ones(shape, dtype=float)
        for target in self.mapped_targets():
            mask *= 1.0 - np.exp(
                -0.5 * ((k_mag - float(target)) / sigma) ** 2
            )
        mask = np.clip(mask, 0.0, 1.0)
        mask[0, 0] = 1.0
        return mask

    def control_metadata(self, shape: tuple[int, int]) -> dict[str, object]:
        mask = self.spectral_mask(shape)
        non_dc = np.ones(shape, dtype=bool)
        non_dc[0, 0] = False
        targets = self.mapped_targets()
        return {
            "mode": self.mode,
            "targets": [float(x) for x in targets],
            "target_mean": float(np.mean(targets)),
            "target_span": float(np.ptp(targets)),
            "narrow_sigma": (
                None
                if self.mode in {"legacy_broad", "noise_only"}
                else float(self.narrow_sigma())
            ),
            "neighbor_cross_talk": self.neighbor_cross_talk,
            "non_dc_mask_mean": float(np.mean(mask[non_dc])),
            "non_dc_mask_median": float(np.median(mask[non_dc])),
            "non_dc_fraction_below_0_9": float(np.mean(mask[non_dc] < 0.9)),
        }


def _sim_for_mode(
    base: ScenarioConfig,
    mode: str,
    seed: int,
    *,
    geometry_enabled: bool,
    Nx: int,
    Ny: int,
    dt_h: float,
    include_clay: bool,
    preseed_rna: bool,
    outdir: str,
) -> UniversalOriginSimulator:
    cfg = deepcopy(base)
    cfg.seed = int(seed)
    if not geometry_enabled:
        cfg.topo_strength = 0.0

    cfg.use_zeta_constraints = mode != "none"
    sim = UniversalOriginSimulator(
        cfg,
        Nx=Nx,
        Ny=Ny,
        dt_h=dt_h,
        outdir=f"{outdir}/seed_{seed}/{mode}",
        include_clay=include_clay,
        preseed_rna=preseed_rna,
    )

    if mode != "none":
        sim.zeta_modulator = ControlledZetaModulator(
            mode=mode,
            lambda_soft=cfg.zeta_lambda_soft,
            sigma_heis=cfg.zeta_sigma_heis,
        )
    return sim


def run_zeta_specificity_controls(
    config: ScenarioConfig,
    *,
    seeds: Sequence[int] | None = None,
    geometry_enabled: bool = True,
    modes: Sequence[str] = MODES,
    Nx: int = 32,
    Ny: int = 32,
    dt_h: float = 0.05,
    hours: float = 120.0,
    include_clay: bool = True,
    preseed_rna: bool = True,
    outdir: str = "outputs_zeta_specificity",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run matched zeta-specificity controls and return endpoints + contrasts."""
    run_modes = [str(m) for m in modes]
    bad = [m for m in run_modes if m not in MODES]
    if bad:
        raise ValueError(f"unsupported modes: {bad}")

    run_seeds = [int(config.seed)] if seeds is None else [int(s) for s in seeds]
    if not run_seeds:
        raise ValueError("at least one seed is required")

    rows: list[dict[str, object]] = []
    for seed in run_seeds:
        sims: dict[str, UniversalOriginSimulator] = {}
        digests: dict[str, str] = {}

        for mode in run_modes:
            sim = _sim_for_mode(
                config,
                mode,
                seed,
                geometry_enabled=geometry_enabled,
                Nx=Nx,
                Ny=Ny,
                dt_h=dt_h,
                include_clay=include_clay,
                preseed_rna=preseed_rna,
                outdir=outdir,
            )
            sim.initialize()
            sims[mode] = sim
            digests[mode] = initial_state_digest(sim)

        if len(set(digests.values())) != 1:
            raise RuntimeError(
                f"seed {seed}: unmatched initial states across controls: {digests}"
            )

        digest = next(iter(digests.values()))
        for mode, sim in sims.items():
            sim.run(
                hours=hours,
                record_interval=max(dt_h, hours),
                verbose=False,
            )
            row: dict[str, object] = {
                "scenario": config.code,
                "seed": seed,
                "mode": mode,
                "geometry_enabled": bool(geometry_enabled),
                "initial_state_digest": digest,
            }
            row.update(_final_metrics(sim))
            rows.append(row)

    runs = pd.DataFrame(rows)
    contrasts: list[dict[str, object]] = []
    for seed, group in runs.groupby("seed", sort=True):
        by_mode = group.set_index("mode")
        if "none" not in by_mode.index:
            raise ValueError("specificity contrasts require mode='none'")
        for metric in METRICS:
            base = float(by_mode.loc["none", metric])
            noise = (
                float(by_mode.loc["noise_only", metric])
                if "noise_only" in by_mode.index
                else float("nan")
            )
            for mode in run_modes:
                value = float(by_mode.loc[mode, metric])
                contrasts.append(
                    {
                        "seed": int(seed),
                        "metric": metric,
                        "mode": mode,
                        "delta_vs_none": value - base,
                        "delta_vs_noise_only": value - noise,
                    }
                )

    return runs, pd.DataFrame(contrasts)
