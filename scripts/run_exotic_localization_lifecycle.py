from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from origins.exobiology.localization_lifecycle import (
    lifecycle_manifest,
    run_matched_localization_lifecycle,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="exotic_localization_lifecycle_artifact")
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    args = parser.parse_args()

    rows = [
        row.as_record()
        for row in run_matched_localization_lifecycle(
            [SCENARIO_C, SCENARIO_D],
            seeds=args.seeds,
            horizon_steps=args.horizon,
            Nx=args.nx,
            Ny=args.ny,
        )
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": lifecycle_manifest(),
        "configuration": {
            "horizon_steps": args.horizon,
            "Nx": args.nx,
            "Ny": args.ny,
            "seeds": args.seeds,
        },
        "row_count": len(rows),
        "rows": rows,
    }

    json_path = outdir / "exotic_localization_lifecycle_v01.json"
    csv_path = outdir / "exotic_localization_lifecycle_v01.csv"
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
