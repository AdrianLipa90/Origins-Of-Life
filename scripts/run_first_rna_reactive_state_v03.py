#!/usr/bin/env python3
"""Run first-RNA reactive-state calibration v0.3."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.first_rna_reactive_state_v03 import (
    ACTIVATED_CLAY_PROFILE,
    CYCLIC_WET_DRY_PROFILE,
    run_reactive_state_calibration,
)


PREREG = "docs/FIRST_RNA_REACTIVE_STATE_PREREG_V0_3.md"


def _profile_dict(profile):
    return {
        "name": profile.name,
        "hours": profile.hours,
        "dt_h": profile.dt_h,
        "initial_units": profile.initial_units,
        "initial_reactive_fraction": profile.initial_reactive_fraction,
        "max_len": profile.max_len,
        "k_activation": profile.k_activation,
        "k_deactivation": profile.k_deactivation,
        "k_nucleation": profile.k_nucleation,
        "k_extension": profile.k_extension,
        "k_hydrolysis": profile.k_hydrolysis,
        "cycle_h": profile.cycle_h,
        "dry_fraction": profile.dry_fraction,
        "dry_extension_multiplier": profile.dry_extension_multiplier,
        "dry_hydrolysis_multiplier": profile.dry_hydrolysis_multiplier,
    }


def main() -> int:
    rows, summary = run_reactive_state_calibration()
    out = Path("first_rna_reactive_state_v03")
    out.mkdir(parents=True, exist_ok=True)

    rows_path = out / "first_rna_reactive_state_v03_rows.csv"
    receipt_path = out / "first_rna_reactive_state_v03_receipt.json"
    rows.to_csv(rows_path, index=False)

    receipt = {
        **summary,
        "preregistration": PREREG,
        "profiles": {
            "cyclic_wet_dry": _profile_dict(CYCLIC_WET_DRY_PROFILE),
            "activated_clay": _profile_dict(ACTIVATED_CLAY_PROFILE),
        },
        "rows_csv": str(rows_path),
        "empirical_anchors": {
            "cyclic_wet_dry_doi": "10.1021/acscentsci.5c00488",
            "activated_clay_doi": "10.1021/ja061782k",
            "long_tail_maldi_doi": "10.1016/j.jasms.2006.05.012",
        },
        "physical_rate_claim": False,
        "predictive_validation_claim": False,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(rows.to_string(index=False))
    print()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
