#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.orbital_formalism import build_orbital_repository_snapshot
from origins.scenarios import ALL_SCENARIOS, SCENARIOS_BY_CODE


def main() -> None:
    parser = argparse.ArgumentParser(description="Export orbital repository manifest")
    parser.add_argument("--out", type=str, default="outputs/orbital_repository_manifest.json")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Scenario codes to include (default: all)")
    args = parser.parse_args()

    if args.scenarios:
        configs = [SCENARIOS_BY_CODE[c] for c in args.scenarios]
    else:
        configs = ALL_SCENARIOS

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    snapshot = build_orbital_repository_snapshot(configs)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"Orbital manifest written to: {args.out}")


if __name__ == "__main__":
    main()
