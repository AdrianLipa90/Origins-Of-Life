#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_CANONICAL_FILES = [
    ROOT / "origins" / "abiogenesis" / "__init__.py",
    ROOT / "origins" / "abiogenesis" / "api.py",
    ROOT / "scripts" / "run_abiogenesis.py",
    ROOT / "docs" / "ABIOGENESIS_PUBLIC_SURFACE.md",
    ROOT / "START_HERE_ORIGINS.md",
]

SUBSTRATE_FILES = [
    ROOT / "origins" / "orbital" / "__init__.py",
    ROOT / "origins" / "simulator" / "universal_orbital.py",
    ROOT / "origins" / "analysis" / "sweep_orbital.py",
]


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_CANONICAL_FILES if not p.exists()]
    missing_substrate = [str(p.relative_to(ROOT)) for p in SUBSTRATE_FILES if not p.exists()]

    readme = ROOT / "README.md"
    readme_status = "missing"
    if readme.exists():
        txt = readme.read_text(encoding="utf-8", errors="ignore")
        if "CMB Intelligence Detection" in txt or "CIEL/0" in txt:
            readme_status = "legacy_framing_detected"
        else:
            readme_status = "repo_aligned"

    report = {
        "canonical_surface_ok": len(missing) == 0,
        "substrate_ok": len(missing_substrate) == 0,
        "missing_canonical_files": missing,
        "missing_substrate_files": missing_substrate,
        "readme_status": readme_status,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0 if report["canonical_surface_ok"] and report["substrate_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
