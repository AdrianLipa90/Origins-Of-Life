# ORIGINS CODE MERGE — DENESTED ZIP SURFACE

Created: 2026-06-09T09:24:23Z

This package recursively unpacks the uploaded Origins/NOEMA archive line, including nested ZIP/TAR archives, and merges code without destructive overwrites.

## Result

- Source containers scanned: 20
- Records including archive members: 7579
- Non-archive records: 7502
- Unique file SHA-256 count: 262
- Archive records found / denested: 77
- Code records: 4533
- Python records: 4065
- Unique code SHA count: 122
- Canonical candidate code files: 527
- Code path conflicts preserved: 4
- Python syntax PASS records: 4065
- Python syntax non-PASS records: 0

## Directory map

- `src_canonical_candidates/` — best candidate for each normalized code path. Selection: highest version score, then larger file size.
- `src_all_unique_by_sha/` — every unique code file by SHA; no overwrites.
- `src_conflict_variants/` — all variants where the same normalized code path had conflicting hashes.
- `docs_paper/` — paper/docs/README/LaTeX/BibTeX material pulled out of nested bundles.
- `noema_manifests/` — CURRENT, manifest, NOEMA, reconstruction and report JSON/YAML/JSONL artifacts.
- `data_results/` — result tables.
- `denested_archive_inventory/` — list of nested archives found.
- `reports/` — CSV/JSON audit reports.

## Important boundary

This is an artifact-grounded merge from uploaded files. It is not a verified external NOEMA CURRENT promotion and does not claim external filesystem access.
