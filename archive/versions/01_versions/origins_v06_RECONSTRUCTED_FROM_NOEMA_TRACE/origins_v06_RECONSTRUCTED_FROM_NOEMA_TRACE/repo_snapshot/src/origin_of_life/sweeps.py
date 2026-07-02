import concurrent.futures
import multiprocessing
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import save_figure
from .simulator import UniversalOriginSimulator
from .utils import ensure_dir


def _topo_sweep_worker(args):
    cfg_copy, s_val, ks, Nx, Ny, dt_h, hours, outdir_cfg = args
    cfg_copy = deepcopy(cfg_copy)
    cfg_copy.topo_strength = float(s_val)
    cfg_copy.k_synthesis = float(ks)
    sim = UniversalOriginSimulator(cfg_copy, Nx=Nx, Ny=Ny, dt_h=dt_h, outdir=outdir_cfg)
    sim.initialize()
    sim.run(hours=hours, record_interval=max(1.0, hours / 20.0), verbose=False)
    return (s_val, ks, int(sim.protocell_count))


def run_topo_param_sweep(all_scenarios, topo_list, synth_list, Nx=64, Ny=64, dt_h=0.05, hours=60.0, outdir="topo_sweep_outputs", enable_multiproc=False, workers=4):
    ensure_dir(outdir)
    results = []
    for cfg in all_scenarios:
        outdir_cfg = os.path.join(outdir, f"scenario_{cfg.code}")
        ensure_dir(outdir_cfg)
        metric_matrix = np.zeros((len(topo_list), len(synth_list)), dtype=float)
        arglist = [(deepcopy(cfg), s_val, ks, Nx, Ny, dt_h, hours, outdir_cfg) for s_val in topo_list for ks in synth_list]
        if enable_multiproc:
            max_workers = min(workers, max(1, multiprocessing.cpu_count()))
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
                for s_val, ks, metric in exe.map(_topo_sweep_worker, arglist):
                    i = topo_list.index(s_val)
                    j = synth_list.index(ks)
                    metric_matrix[i, j] = metric
        else:
            for item in arglist:
                s_val, ks, metric = _topo_sweep_worker(item)
                i = topo_list.index(s_val)
                j = synth_list.index(ks)
                metric_matrix[i, j] = metric

        np.savez_compressed(os.path.join(outdir_cfg, "metric_matrix.npz"), topo_list=np.array(topo_list), synth_list=np.array(synth_list), metric=metric_matrix)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(metric_matrix, origin="lower", aspect="auto")
        ax.set_xticks(range(len(synth_list)))
        ax.set_xticklabels([f"{v:.4f}" for v in synth_list], rotation=45)
        ax.set_yticks(range(len(topo_list)))
        ax.set_yticklabels([f"{v:.3f}" for v in topo_list])
        fig.colorbar(im, ax=ax, label="final protocell count")
        ax.set_xlabel("k_synthesis")
        ax.set_ylabel("topo_strength")
        ax.set_title(f"Topo x k_synthesis — scenario {cfg.code}")
        save_figure(fig, os.path.join(outdir_cfg, "topo_vs_synth_heatmap.png"))
        results.append({"scenario": cfg.code, "metric_matrix_file": os.path.join(outdir_cfg, "metric_matrix.npz"), "heatmap": os.path.join(outdir_cfg, "topo_vs_synth_heatmap.png")})

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(outdir, "topo_sweep_summary.csv"), index=False)
    return df_res
