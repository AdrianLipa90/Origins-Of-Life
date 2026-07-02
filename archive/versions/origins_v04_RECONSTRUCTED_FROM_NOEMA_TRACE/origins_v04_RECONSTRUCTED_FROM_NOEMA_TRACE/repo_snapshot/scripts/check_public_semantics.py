#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PUBLIC_DOCS = [
    ROOT / "START_HERE_ORIGINS.md",
    ROOT / "docs" / "ABIOGENESIS_PUBLIC_SURFACE.md",
    ROOT / "docs" / "ABIOGENESIS_WORKFLOWS.md",
    ROOT / "docs" / "REPO_POSITIONING_AND_ADOPTION.md",
]

FORBIDDEN_PUBLIC_PATTERNS = [
    r"\borigins\.orbital\b",
    r"\bOrbitalUniversalOriginSimulator\b",
    r"\brun_simulation_native_orbital\.py\b",
    r"\brun_sweep_native_orbital\.py\b",
]

ALLOWED_CONTEXT_PATTERNS = [
    r"internal",
    r"substrate",
    r"implementation",
    r"legacy",
]


def main() -> int:
    violations = []
    for doc in PUBLIC_DOCS:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_PUBLIC_PATTERNS:
            for match in re.finditer(pattern, text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                if line_end == -1:
                    line_end = len(text)
                line = text[line_start:line_end]
                if not any(re.search(ctx, line, flags=re.IGNORECASE) for ctx in ALLOWED_CONTEXT_PATTERNS):
                    violations.append({
                        "file": str(doc.relative_to(ROOT)),
                        "pattern": pattern,
                        "line": line.strip(),
                    })

    report = {
        "ok": len(violations) == 0,
        "violations": violations,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
