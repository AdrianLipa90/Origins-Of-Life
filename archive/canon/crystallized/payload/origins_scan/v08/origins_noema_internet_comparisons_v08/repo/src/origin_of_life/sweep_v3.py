import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .grid import laplacian
from .plotting import save_figure
from .utils import ensure_dir


def sun_envelope_window(hour, mu=1.0, day_fraction=0.5):
    h = hour % 24.0
    half = 12.0 * day_fraction
    d = abs(h - 12.0)
    if d > half or half <= 0:
        return 0.0
    return float(mu * 0.5 * (1.0 + np.cos(np.pi * d / half)))


def tide_semidiurnal(hour):
    return np.sin(2 * np.pi * hour / 12.42)


def run_once_sweep_v3(Nx=96, Ny=96, hours=30.0, dt_h=0.05, seed=42, EM_amp0=0.20, mu=1.0, day_fraction=0.50):
    rng = np.random.default_rng(seed)
    D_U0, D_P0, D_L0, D_M0, D_C0 = 0.35, 0.12, 0.08, 0.015, 0.25
    k_photo0 = 0.35
    k_poly0 = 0.08
    k_hyd = 0.04
    k_lip0 = 0.05
    k_degL = 0.03
    k_agg0 = 0.35
    k_break0 = 0.010
    k_red_gen0 = 0.10
    k_red_loss = 0.06
    alpha_c = 0.474812
    beta_s = 0.856234
    PhiI0 = 0.02

    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    grad = 0.6 * (X + 1.0) / 2.0
    speckle = 0.15 * rng.standard_normal((Nx, Ny))
    hotspots = 0.7 * np.exp(-((X - 0.35) ** 2 + (Y + 0.10) ** 2) / 0.03) + 0.5 * np.exp(-((X + 0.45) ** 2 + (Y - 0.25) ** 2) / 0.02)
    light_field = np.clip(0.6 + grad + speckle + hotspots, 0.0, 2.0)
    EM_pattern = np.sin(2 * np.pi * 2 * X) * np.sin(2 * np.pi * 3 * Y)

    shore = (Y > -0.2) & (Y < -0.1)
    patches = rng.random((Nx, Ny)) < 0.06
    catalyst = (shore | patches).astype(float)
    catalyst = np.clip(catalyst + 0.5 * np.exp(-((X + 0.2) ** 2 + (Y + 0.5) ** 2) / 0.02), 0, 1)

    U = 0.9 + 0.1 * rng.random((Nx, Ny))
    P = 0.05 * rng.random((Nx, Ny))
    L = 0.25 * rng.random((Nx, Ny))
    M = 0.04 * rng.random((Nx, Ny))
    C = 0.05 * rng.random((Nx, Ny))

    steps = int(hours / dt_h)
    area_series = []
    for n in range(steps):
        t_h = n * dt_h
        S = sun_envelope_window(t_h, mu=mu, day_fraction=day_fraction)
        Nflag = 1.0 - (1.0 if S > 0 else 0.0)
        Tide = tide_semidiurnal(t_h)
        TideAbs = abs(Tide)

        k_photo = k_photo0 * (1 + 1.2 * S)
        k_lip = k_lip0 * (1 + 0.2 * S)
        k_agg = k_agg0 * (1 + 0.10 * TideAbs)
        k_break = k_break0 * (1 + 0.60 * TideAbs)
        k_poly = k_poly0 * (1 + 0.10 * S + 0.05 * Tide)
        shore_factor = 1 + 0.50 * max(0.0, -Tide)
        k_poly_map = k_poly * (1 + shore_factor * shore.astype(float))
        k_red_gen = k_red_gen0 * (1 + 0.3 * Nflag + 0.1 * S)

        D_U = D_U0 * (1 + 0.15 * TideAbs)
        D_P = D_P0 * (1 + 0.10 * TideAbs)
        D_L = D_L0 * (1 + 0.10 * TideAbs)
        D_M = D_M0 * (1 + 0.20 * TideAbs)
        D_C = D_C0 * (1 + 0.10 * TideAbs)

        EM_env = EM_amp0 * (1 + 0.35 * Nflag + 0.05 * S) * EM_pattern
        photofactor = 1.0 + 0.8 * light_field * (0.2 + 0.8 * S) + 0.25 * EM_env

        r_photo = k_photo * photofactor * U
        r_poly = k_poly_map * (1.0 + 0.25 * beta_s * C) * U * (P + 0.05)
        r_lip = k_lip * (1.0 + 0.4 * catalyst) * U

        dU = -r_photo - r_poly - r_lip + k_hyd * P
        dP = r_photo + r_poly - k_hyd * P
        dL = r_lip - k_degL * L - k_agg * (L * L) + k_break * M
        curvature = 1.0 + 0.6 * np.tanh(EM_env)
        dM = k_agg * (L * L) * curvature * (1.0 + PhiI0) - k_break * M
        dC = k_red_gen * (0.5 * (0.2 + 0.8 * S) * light_field + 0.4 * np.abs(EM_env)) - k_red_loss * C

        U += dt_h * (D_U * laplacian(U) + dU)
        P += dt_h * (D_P * laplacian(P) + dP)
        L += dt_h * (D_L * laplacian(L) + dL)
        M += dt_h * (D_M * laplacian(M) + dM)
        C += dt_h * (D_C * laplacian(C) + dC)

        for Z in (U, P, L, M, C):
            Z += (alpha_c * 1e-3) * rng.standard_normal(Z.shape)
            np.clip(Z, 0.0, None, out=Z)

        if n % int(1.0 / dt_h) == 0 or n == steps - 1:
            thresh = max(0.08, np.percentile(M, 90))
            binM = M > thresh
            area_series.append(float(binM.mean()))

    return {
        "avgP_final": float(P.mean()),
        "avgM_final": float(M.mean()),
        "vesicle_area_mean": float(np.mean(area_series)),
        "vesicle_area_max": float(np.max(area_series)),
    }


def run_sweep_v3_and_save(outdir="sweep_v3_outputs", Nx=96, Ny=96, dt_h=0.05):
    ensure_dir(outdir)
    EM_list = [0.10, 0.20, 0.30, 0.40]
    mu_list = [0.5, 0.75, 1.0]
    dayf_list = [0.33, 0.50, 0.67]

    rows = []
    combo_id = 0
    for EM_amp0 in EM_list:
        for mu in mu_list:
            for dayf in dayf_list:
                combo_id += 1
                metrics = run_once_sweep_v3(EM_amp0=EM_amp0, mu=mu, day_fraction=dayf, hours=30.0, dt_h=dt_h, seed=4748 + combo_id, Nx=Nx, Ny=Ny)
                rows.append({"EM_amp0": EM_amp0, "mu": mu, "day_fraction": dayf, **metrics})
    df = pd.DataFrame(rows)
    csv_fn = os.path.join(outdir, "prebiotic_sweep_v3_results.csv")
    df.to_csv(csv_fn, index=False)

    for dayf in dayf_list:
        sub = df[df["day_fraction"] == dayf]
        pivot_area = sub.pivot(index="EM_amp0", columns="mu", values="vesicle_area_mean")
        pivot_M = sub.pivot(index="EM_amp0", columns="mu", values="avgM_final")

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(pivot_area.values, origin="lower", aspect="auto")
        ax.set_xticks(range(len(pivot_area.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot_area.columns])
        ax.set_yticks(range(len(pivot_area.index)))
        ax.set_yticklabels([f"{r:.2f}" for r in pivot_area.index])
        fig.colorbar(im, ax=ax, label="mean high-M area")
        ax.set_xlabel("mu = cos(zenith)")
        ax.set_ylabel("EM_amp0")
        ax.set_title(f"Vesicle yield — day_fraction={dayf:.2f}")
        save_figure(fig, os.path.join(outdir, f"heatmap_area_dayf_{int(dayf * 100)}.png"))

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(pivot_M.values, origin="lower", aspect="auto")
        ax.set_xticks(range(len(pivot_M.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot_M.columns])
        ax.set_yticks(range(len(pivot_M.index)))
        ax.set_yticklabels([f"{r:.2f}" for r in pivot_M.index])
        fig.colorbar(im, ax=ax, label="avgM_final")
        ax.set_xlabel("mu = cos(zenith)")
        ax.set_ylabel("EM_amp0")
        ax.set_title(f"Final membrane fraction — day_fraction={dayf:.2f}")
        save_figure(fig, os.path.join(outdir, f"heatmap_M_dayf_{int(dayf * 100)}.png"))

    return df
