from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from origins.exobiology import (
    create_exotic_candidate_simulator,
)
from origins.exobiology.source_geometry import diagnose_source_geometry
from origins.scenarios import SCENARIO_C, SCENARIO_D


SCHEMA = "ORIGINS_EXOTIC_SOURCE_GEOMETRY_SCAN_V0_1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="exotic_source_geometry_artifact")
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 300, 1000, 3000])
    args = parser.parse_args()

    checkpoints = tuple(sorted(set(args.steps)))
    if not checkpoints or checkpoints[0] < 0:
        parser.error("--steps must contain non-negative checkpoints")
    if args.nx <= 1 or args.ny <= 1:
        parser.error("--nx/--ny must be > 1")

    rows: list[dict[str, object]] = []
    for scenario in (SCENARIO_C, SCENARIO_D):
        for seed in args.seeds:
            sim = create_exotic_candidate_simulator(
                scenario,
                Nx=args.nx,
                Ny=args.ny,
                seed=seed,
            )
            sim.initialize()
            previous = 0
            for checkpoint in checkpoints:
                if checkpoint > previous:
                    sim.run(checkpoint - previous)
                previous = checkpoint
                row = diagnose_source_geometry(sim).as_record()
                row["seed"] = int(seed)
                row["step"] = int(checkpoint)
                rows.append(row)

    payload = {
        "schema": SCHEMA,
        "ranking_allowed": False,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "physical_binding": "OPEN",
        "configuration": {
            "Nx": args.nx,
            "Ny": args.ny,
            "seeds": list(args.seeds),
            "steps": list(checkpoints),
        },
        "row_count": len(rows),
        "rows": rows,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "exotic_source_geometry_v01.json"
    csv_path = outdir / "exotic_source_geometry_v01.csv"

    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "schema": SCHEMA,
        "row_count": len(rows),
        "json": str(json_path),
        "csv": str(csv_path),
        "ranking_allowed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
