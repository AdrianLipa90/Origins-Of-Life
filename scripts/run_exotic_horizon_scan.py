from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from origins.exobiology.horizon_scan import (
    DEFAULT_HORIZONS,
    HORIZON_SCAN_SCHEMA,
    horizon_scan_manifest,
    run_matched_exotic_horizon_scan,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="exotic_horizon_artifact")
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    args = parser.parse_args()

    results = run_matched_exotic_horizon_scan(
        [SCENARIO_C, SCENARIO_D],
        seeds=args.seeds,
        horizons=args.horizons,
        Nx=args.nx,
        Ny=args.ny,
    )
    rows = [r.as_record() for r in results]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": HORIZON_SCAN_SCHEMA,
        "manifest": horizon_scan_manifest(),
        "configuration": {
            "Nx": args.nx,
            "Ny": args.ny,
            "seeds": args.seeds,
            "horizons": args.horizons,
        },
        "row_count": len(rows),
        "rows": rows,
    }

    json_path = outdir / "exotic_horizon_scan_v01.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = outdir / "exotic_horizon_scan_v01.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "schema": HORIZON_SCAN_SCHEMA,
        "row_count": len(rows),
        "json": str(json_path),
        "csv": str(csv_path),
        "ranking_allowed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
