from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from origins.exobiology.hydrocarbon_island_ancestry import (
    ancestry_manifest,
    run_hydrocarbon_island_ancestry_case,
)
from origins.scenarios import SCENARIO_D


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="hydrocarbon_island_ancestry_artifact")
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    args = parser.parse_args()

    rows = [
        run_hydrocarbon_island_ancestry_case(
            SCENARIO_D,
            seed=seed,
            horizon_steps=args.horizon,
            Nx=args.nx,
            Ny=args.ny,
        ).as_record()
        for seed in args.seeds
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": ancestry_manifest(),
        "configuration": {
            "horizon_steps": args.horizon,
            "Nx": args.nx,
            "Ny": args.ny,
            "seeds": args.seeds,
        },
        "row_count": len(rows),
        "rows": rows,
    }

    json_path = outdir / "hydrocarbon_island_ancestry_v01.json"
    csv_path = outdir / "hydrocarbon_island_ancestry_v01.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["transitions"] = json.dumps(row["transitions"], sort_keys=True)
        flat_rows.append(flat)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps({
        "schema": payload["manifest"]["schema"],
        "row_count": len(rows),
        "json": str(json_path),
        "csv": str(csv_path),
        "diagnostic_only": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
