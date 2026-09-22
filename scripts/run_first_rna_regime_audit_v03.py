#!/usr/bin/env python3
"""Emit the First-RNA reaction-regime v0.3 representation receipt."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.chemistry.rna_polymerization_regimes import (
    CAIMI_2025_AUGC_PH11,
    QT45_FUNCTIONAL_ANCHOR,
    SONG_2024_UMP_HOT_ACID,
    representation_gate_receipt,
)


def _anchor_to_json(anchor):
    return {
        "regime_id": anchor.regime_id.value,
        "doi": anchor.doi,
        "monomer_state": anchor.monomer_state.value,
        "interface": anchor.interface.value,
        "temperature_C": anchor.temperature_C,
        "pH_window": [anchor.pH_min, anchor.pH_max],
        "concentration_mM_window": [
            anchor.concentration_mM_min,
            anchor.concentration_mM_max,
        ],
        "cycle_window": [anchor.cycles_min, anchor.cycles_max],
        "observables": dict(anchor.observables),
        "limitations": list(anchor.limitations),
    }


def main() -> int:
    receipt = representation_gate_receipt()
    receipt["empirical_anchors"] = [
        _anchor_to_json(CAIMI_2025_AUGC_PH11),
        _anchor_to_json(SONG_2024_UMP_HOT_ACID),
    ]
    receipt["functional_anchor"] = {
        "name": QT45_FUNCTIONAL_ANCHOR.name,
        "doi": QT45_FUNCTIONAL_ANCHOR.doi,
        "length_nt": QT45_FUNCTIONAL_ANCHOR.length_nt,
        "observables": dict(QT45_FUNCTIONAL_ANCHOR.observables),
        "warning": QT45_FUNCTIONAL_ANCHOR.warning,
    }

    out = Path("first_rna_regime_v03")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "first_rna_reaction_regimes_v03_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
