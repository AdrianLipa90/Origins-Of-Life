from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from origins.exobiology.benchmark import (
    ABLATIONS,
    BENCHMARK_SCHEMA,
    COMPARISON_SCOPE,
    benchmark_manifest,
    run_matched_exotic_relational_benchmark,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        profile = str(row["profile"])
        ablation = str(row["ablation"])
        bucket = grouped.setdefault(profile, {}).setdefault(
            ablation,
            {
                "n": 0.0,
                "material_residual_abs_sum": 0.0,
                "energy_throughput_integral_sum": 0.0,
                "information_mass_sum": 0.0,
                "boundary_mass_sum": 0.0,
                "candidate_compartment_count_sum": 0.0,
                "trait_mean_sum": 0.0,
                "trait_variance_sum": 0.0,
            },
        )
        bucket["n"] += 1.0
        for key in tuple(bucket):
            if key == "n":
                continue
            source = key.removesuffix("_sum")
            bucket[key] += float(row[source])

    out: dict[str, dict[str, dict[str, float]]] = {}
    for profile, ablations in grouped.items():
        out[profile] = {}
        for ablation, bucket in ablations.items():
            n = bucket["n"]
            out[profile][ablation] = {
                key.removesuffix("_sum") + "_mean": value / n
                for key, value in bucket.items()
                if key != "n"
            }
            out[profile][ablation]["n"] = n
    return {
        "aggregation": "MEAN_WITHIN_PROFILE_AND_ABLATION_ONLY",
        "cross_profile_ranking": False,
        "profiles": out,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="exotic_benchmark_artifact")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    args = parser.parse_args()

    results = run_matched_exotic_relational_benchmark(
        [SCENARIO_C, SCENARIO_D],
        seeds=args.seeds,
        steps=args.steps,
        Nx=args.nx,
        Ny=args.ny,
        ablations=ABLATIONS,
    )
    rows = [r.as_record() for r in results]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": BENCHMARK_SCHEMA,
        "comparison_scope": COMPARISON_SCOPE,
        "manifest": benchmark_manifest(),
        "configuration": {
            "steps": args.steps,
            "Nx": args.nx,
            "Ny": args.ny,
            "seeds": args.seeds,
            "ablations": list(ABLATIONS),
        },
        "row_count": len(rows),
        "summary": _summary(rows),
        "rows": rows,
    }

    json_path = outdir / "exotic_relational_benchmark_v01.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = outdir / "exotic_relational_benchmark_v01.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "schema": BENCHMARK_SCHEMA,
        "row_count": len(rows),
        "json": str(json_path),
        "csv": str(csv_path),
        "ranking_allowed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
