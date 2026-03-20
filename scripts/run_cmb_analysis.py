#!/usr/bin/env python3
"""
Run CIEL/0 CMB modulation analysis.

Examples
--------
    python scripts/run_cmb_analysis.py --shape 256 256 --outdir cmb_outputs/
    python scripts/run_cmb_analysis.py --threshold 0.6
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from origins.cosmology import TelescopeDataModulator


def main():
    parser = argparse.ArgumentParser(description="CIEL/0 CMB modulation analysis")
    parser.add_argument("--shape", nargs=2, type=int, default=[256, 256],
                        metavar=("NX", "NY"))
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Intelligence signature detection threshold")
    parser.add_argument("--outdir", type=str, default="cmb_outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    shape = tuple(args.shape)

    mod = TelescopeDataModulator()
    print(f"Computing CIEL/0 modulation field {shape[0]}×{shape[1]}…")
    field = mod.compute_unified_modulation(data_shape=shape)

    np.save(os.path.join(args.outdir, "modulation_field.npy"), field)

    sigs = mod.detect_intelligence_signatures(field, threshold=args.threshold)
    print("\nIntelligence signature analysis:")
    for k, v in sigs.items():
        print(f"  {k}: {v}")

    # Visualise if matplotlib available
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(field, origin="lower", aspect="auto", cmap="inferno")
        fig.colorbar(im, ax=ax, label="Modulation M(x,y)")
        ax.set_title("CIEL/0 Unified CMB Modulation Field")
        fig.tight_layout()
        path = os.path.join(args.outdir, "modulation_field.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\nHeatmap saved: {path}")
    except ImportError:
        pass

    print(f"\nOutputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
