from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from origins.exobiology.morphogenesis_factorial import (
    factorial_manifest,
    run_profile_factorial,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="exotic_morphogenesis_factorial_artifact")
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    parser.add_argument("--ammonia-horizon", type=int, default=3000)
    parser.add_argument("--hydrocarbon-horizon", type=int, default=10000)
    args = parser.parse_args()

    ammonia = run_profile_factorial(
        SCENARIO_C,
        seeds=args.seeds,
        horizon_steps=args.ammonia_horizon,
        Nx=args.nx,
        Ny=args.ny,
    )
    hydrocarbon = run_profile_factorial(
        SCENARIO_D,
        seeds=args.seeds,
        horizon_steps=args.hydrocarbon_horizon,
        Nx=args.nx,
        Ny=args.ny,
    )
    rows = [row.as_record() for row in [*ammonia, *hydrocarbon]]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": {
            **factorial_manifest(),
            "profile_horizons": {
                "AMMONIA_CANDIDATE": args.ammonia_horizon,
                "HYDROCARBON_CANDIDATE": args.hydrocarbon_horizon,
            },
            "horizon_rationale": (
                "Use previously validated lifecycle windows that contain the "
                "observed transition regime for each profile; no model parameter "
                "or detection threshold is changed."
            ),
        },
        "configuration": {
            "Nx": args.nx,
            "Ny": args.ny,
            "seeds": args.seeds,
        },
        "row_count": len(rows),
        "rows": rows,
    }

    json_path = outdir / "exotic_morphogenesis_factorial_v03.json"
    csv_path = outdir / "exotic_morphogenesis_factorial_v03.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "schema": payload["manifest"]["schema"],
        "row_count": len(rows),
        "json": str(json_path),
        "csv": str(csv_path),
        "ranking_allowed": False,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
