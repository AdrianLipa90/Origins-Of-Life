"""Preregistered zeta spectral-specificity experiment v0.2.

The design is frozen in docs/ZETA_SPECTRAL_SPECIFICITY_PREREG_V0_2.md.
Do not change seeds, endpoints, control definitions, or verdict thresholds after
examining v0.2 endpoint results.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from typing import Sequence

import numpy as np
import pandas as pd

from ..scenarios import ScenarioConfig
from ..simulator import UniversalOriginSimulator
from ..topology.constraints import ZetaRiemannModulator
from .ablation import _final_metrics, initial_state_digest


V02_MODES = (
    "none",
    "noise_only",
    "narrow_zeta",
    "narrow_reflected",
    "narrow_shifted",
    "narrow_equispaced",
    "histogram_permuted",
)

PRIMARY_METRIC = "n_protocells"
SECONDARY_METRICS = (
    "protocell_area_pixels",
    "polymer_threshold_pixels",
    "mean_R",
    "max_R",
    "n_polymers",
    "nucleotide_material_total",
)

PLACEBO_MODES = (
    "narrow_reflected",
    "narrow_shifted",
    "narrow_equispaced",
    "histogram_permuted",
)

PERMUTATION_SEED = 20260922
SHIFT_CYCLES_PER_SAMPLE = -0.04
NEIGHBOR_CROSS_TALK = 0.01


def _base_targets(zeros: Sequence[complex]) -> np.ndarray:
    targets = np.array(
        [
            ZetaRiemannModulator._normalized_target(abs(complex(z).imag))
            for z in zeros
        ],
        dtype=float,
    )
    if targets.size < 2:
        raise ValueError("v0.2 specificity controls require at least two targets")
    return targets


def _narrow_sigma(targets: np.ndarray) -> float:
    ordered = np.sort(np.asarray(targets, dtype=float))
    dmin = float(np.min(np.diff(ordered)))
    if dmin <= 0.0:
        raise ValueError("targets must be distinct")
    return dmin / math.sqrt(
        2.0 * math.log(1.0 / NEIGHBOR_CROSS_TALK)
    )


def _radial_frequency(shape: tuple[int, int]) -> np.ndarray:
    Nx, Ny = shape
    if Nx <= 0 or Ny <= 0:
        raise ValueError("shape dimensions must be positive")
    kx = np.fft.fftfreq(Nx)
    ky = np.fft.fftfreq(Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return np.sqrt(KX**2 + KY**2)


def _narrow_mask(
    shape: tuple[int, int],
    targets: np.ndarray,
    sigma: float,
) -> np.ndarray:
    k_mag = _radial_frequency(shape)
    mask = np.ones(shape, dtype=float)
    for target in np.asarray(targets, dtype=float):
        mask *= 1.0 - np.exp(
            -0.5 * ((k_mag - float(target)) / float(sigma)) ** 2
        )
    mask = np.clip(mask, 0.0, 1.0)
    mask[0, 0] = 1.0
    return mask


def _hermitian_classes(shape: tuple[int, int]) -> dict[int, list[list[tuple[int, int]]]]:
    Nx, Ny = shape
    visited: set[tuple[int, int]] = set()
    groups: dict[int, list[list[tuple[int, int]]]] = {1: [], 2: []}
    for i in range(Nx):
        for j in range(Ny):
            idx = (i, j)
            if idx == (0, 0) or idx in visited:
                continue
            neg = ((-i) % Nx, (-j) % Ny)
            members = sorted(set((idx, neg)))
            for member in members:
                visited.add(member)
            groups[len(members)].append(members)
    return groups


def hermitian_histogram_permutation(
    base_mask: np.ndarray,
    *,
    seed: int = PERMUTATION_SEED,
) -> np.ndarray:
    """Permute a real mask while preserving Hermitian symmetry and histogram.

    Values move among Fourier-bin conjugacy classes of the same cardinality.
    This preserves every non-DC mask value exactly while changing where those
    attenuations occur. DC remains one.
    """
    base = np.asarray(base_mask, dtype=float)
    if base.ndim != 2:
        raise ValueError("base_mask must be 2-D")
    if not np.isfinite(base).all():
        raise FloatingPointError("base_mask contains NaN/Inf")

    out = np.empty_like(base)
    out[0, 0] = base[0, 0]
    rng = np.random.default_rng(int(seed))
    groups = _hermitian_classes(base.shape)

    for size, classes in groups.items():
        if not classes:
            continue
        values = np.array(
            [base[members[0]] for members in classes],
            dtype=float,
        )
        permuted = values[rng.permutation(len(values))]
        for members, value in zip(classes, permuted):
            for idx in members:
                out[idx] = value

    if not np.array_equal(
        np.sort(base.ravel()[1:]),
        np.sort(out.ravel()[1:]),
    ):
        raise RuntimeError("histogram permutation failed exact multiset preservation")

    Nx, Ny = base.shape
    for i in range(Nx):
        for j in range(Ny):
            if out[i, j] != out[(-i) % Nx, (-j) % Ny]:
                raise RuntimeError("histogram permutation broke Hermitian symmetry")
    return out


class ZetaSpecificityV02Modulator(ZetaRiemannModulator):
    """Frozen v0.2 spectral-control operator."""

    def __init__(
        self,
        *,
        mode: str,
        zeros: list | None = None,
        lambda_soft: float = 5.0,
        sigma_heis: float = 0.001,
    ):
        if mode not in V02_MODES[1:]:
            raise ValueError(f"unsupported v0.2 mode: {mode!r}")
        super().__init__(
            zeros=zeros,
            lambda_soft=lambda_soft,
            sigma_heis=sigma_heis,
        )
        self.mode = mode

    def target_set(self) -> np.ndarray | None:
        base = _base_targets(self.zeros)
        if self.mode in {"noise_only", "histogram_permuted"}:
            return None
        if self.mode == "narrow_zeta":
            return base.copy()
        if self.mode == "narrow_reflected":
            return 2.0 * float(np.mean(base)) - base
        if self.mode == "narrow_shifted":
            shifted = base + SHIFT_CYCLES_PER_SAMPLE
            if float(np.min(shifted)) <= 0.0:
                raise ValueError("frozen shifted targets left the positive radial band")
            return shifted
        if self.mode == "narrow_equispaced":
            return np.linspace(float(np.min(base)), float(np.max(base)), len(base))
        raise RuntimeError(f"unhandled mode {self.mode!r}")

    def spectral_mask(self, shape: tuple[int, int]) -> np.ndarray:
        if self.mode == "noise_only":
            mask = np.ones(shape, dtype=float)
            mask[0, 0] = 1.0
            return mask

        base = _base_targets(self.zeros)
        sigma = _narrow_sigma(base)

        if self.mode == "histogram_permuted":
            zeta_mask = _narrow_mask(shape, base, sigma)
            return hermitian_histogram_permutation(
                zeta_mask,
                seed=PERMUTATION_SEED,
            )

        targets = self.target_set()
        if targets is None:
            raise RuntimeError("spectral mode requires a target set")
        return _narrow_mask(shape, targets, sigma)

    def mask_diagnostics(self, shape: tuple[int, int]) -> dict[str, object]:
        mask = self.spectral_mask(shape)
        zeta_mask = _narrow_mask(
            shape,
            _base_targets(self.zeros),
            _narrow_sigma(_base_targets(self.zeros)),
        )
        non_dc = np.ones(shape, dtype=bool)
        non_dc[0, 0] = False
        x = mask[non_dc].ravel()
        z = zeta_mask[non_dc].ravel()
        target_set = self.target_set()

        if np.std(x) == 0.0 or np.std(z) == 0.0:
            corr = 1.0 if np.array_equal(x, z) else 0.0
        else:
            corr = float(np.corrcoef(x, z)[0, 1])

        return {
            "mode": self.mode,
            "targets": None if target_set is None else [float(v) for v in target_set],
            "narrow_sigma": float(_narrow_sigma(_base_targets(self.zeros))),
            "non_dc_mean": float(np.mean(x)),
            "non_dc_median": float(np.median(x)),
            "non_dc_fraction_below_0_9": float(np.mean(x < 0.9)),
            "correlation_with_narrow_zeta": corr,
            "mae_vs_narrow_zeta": float(np.mean(np.abs(x - z))),
            "mask_sha256": hashlib.sha256(
                np.ascontiguousarray(mask, dtype=np.float64).tobytes()
            ).hexdigest(),
            "sorted_non_dc_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(np.sort(x), dtype=np.float64).tobytes()
            ).hexdigest(),
            "histogram_exactly_matches_narrow_zeta": bool(
                np.array_equal(np.sort(x), np.sort(z))
            ),
        }


def _simulator(
    base: ScenarioConfig,
    mode: str,
    seed: int,
    *,
    Nx: int,
    Ny: int,
    dt_h: float,
    outdir: str,
) -> UniversalOriginSimulator:
    cfg = deepcopy(base)
    cfg.seed = int(seed)
    cfg.use_zeta_constraints = mode != "none"

    sim = UniversalOriginSimulator(
        cfg,
        Nx=Nx,
        Ny=Ny,
        dt_h=dt_h,
        outdir=f"{outdir}/seed_{seed}/{mode}",
        include_clay=True,
        preseed_rna=True,
    )
    if mode != "none":
        sim.zeta_modulator = ZetaSpecificityV02Modulator(
            mode=mode,
            lambda_soft=cfg.zeta_lambda_soft,
            sigma_heis=cfg.zeta_sigma_heis,
        )
    return sim


def run_zeta_specificity_v02(
    config: ScenarioConfig,
    *,
    seeds: Sequence[int],
    Nx: int = 32,
    Ny: int = 32,
    dt_h: float = 0.05,
    hours: float = 120.0,
    modes: Sequence[str] = V02_MODES,
    outdir: str = "outputs_zeta_specificity_v02",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the frozen v0.2 matched-control design."""
    run_seeds = [int(seed) for seed in seeds]
    if not run_seeds:
        raise ValueError("at least one seed is required")
    bad_seed = [seed for seed in run_seeds if seed < 201 or seed > 220]
    if bad_seed:
        raise ValueError(
            f"v0.2 preregistration permits only seeds 201..220, got {bad_seed}"
        )
    run_modes = [str(mode) for mode in modes]
    bad_modes = [mode for mode in run_modes if mode not in V02_MODES]
    if bad_modes:
        raise ValueError(f"unsupported v0.2 modes: {bad_modes}")
    if Nx != 32 or Ny != 32 or dt_h != 0.05 or hours != 120.0:
        raise ValueError(
            "v0.2 preregistration freezes grid=32x32, dt_h=0.05, hours=120"
        )

    rows: list[dict[str, object]] = []
    for seed in run_seeds:
        sims: dict[str, UniversalOriginSimulator] = {}
        digests: dict[str, str] = {}
        for mode in run_modes:
            sim = _simulator(
                config,
                mode,
                seed,
                Nx=Nx,
                Ny=Ny,
                dt_h=dt_h,
                outdir=outdir,
            )
            sim.initialize()
            sims[mode] = sim
            digests[mode] = initial_state_digest(sim)

        if len(set(digests.values())) != 1:
            raise RuntimeError(
                f"seed {seed}: unmatched v0.2 initial states: {digests}"
            )
        digest = next(iter(digests.values()))

        for mode, sim in sims.items():
            sim.run(
                hours=hours,
                record_interval=hours,
                verbose=False,
            )
            row: dict[str, object] = {
                "scenario": config.code,
                "seed": seed,
                "mode": mode,
                "initial_state_digest": digest,
            }
            row.update(_final_metrics(sim))
            rows.append(row)

    runs = pd.DataFrame(rows)

    diagnostics: list[dict[str, object]] = []
    for mode in run_modes:
        if mode == "none":
            diagnostics.append(
                {
                    "mode": mode,
                    "spectral_operator": False,
                    "noise": False,
                }
            )
        else:
            diag = ZetaSpecificityV02Modulator(
                mode=mode,
                lambda_soft=config.zeta_lambda_soft,
                sigma_heis=config.zeta_sigma_heis,
            ).mask_diagnostics((Nx, Ny))
            diag["spectral_operator"] = mode != "noise_only"
            diag["noise"] = config.zeta_sigma_heis > 0.0
            diagnostics.append(diag)

    return runs, pd.DataFrame(diagnostics)
