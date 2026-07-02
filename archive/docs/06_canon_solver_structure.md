# Chapter 6: Canon Solver Structure

## 6.1 Purpose of 03_canon_solver

The `03_canon_solver` folder collects the clean, reference-grade solver components — code merges, surface anchors, crystallized search results, and reconstructed full traces. Unlike `01_versions` (which preserves historical snapshots), `03_canon_solver` contains the consolidated, deduplicated, and cleanly organized solver core.

## 6.2 Archives in 03_canon_solver

| Archive | Size | Files | Content |
|---------|------|-------|---------|
| `ORIGINS_CODE_MERGE_DENESTED_20260609` | 3.7 MB | 1416 | Code merge with flattened directory structure |
| `ORIGINS_SURFACE_ANCHOR_PULL_20260609` | 5.7 MB | — | NOEMA surface anchor — stable reference state |
| `origins_FULL_CRYSTALLIZED_NOEMA_LINEAGE_v03_to_v10` | 1.0 MB | — | Complete crystallized lineage — all versions consolidated |
| `origins_LOCAL_NOEMA_ALL_FOUND_REPACK_v03_to_v10` | 1.9 MB | — | All local NOEMA traces repacked |
| `origins_NOEMA_PROJECT_SEARCH_CRYSTALLIZED_v03_to_v10_FULL` | 2.3 MB | — | Crystallized project search — full index |
| `origins_RECONSTRUCTED_FROM_NOEMA_TRACE_v03_to_v10_FULL` | 6.3 MB | — | Full reconstruction bundle — all v03-v10 in one archive |
| `origins_noema_internet_comparisons_v08_READY` | 479 KB | — | External validation bundle — v08 ready state |

## 6.3 Key Canon Files

### Reconstruction Bundle Manifest
`origins_RECONSTRUCTED_FROM_NOEMA_TRACE_v03_to_v10_FULL/RECONSTRUCTION_BUNDLE_MANIFEST.json`

Contains SHA256 hashes and provenance metadata for all reconstructed versions. This is the authoritative index for determining which files are byte-exact vs trace-reconstructed.

### Code Merge
`ORIGINS_CODE_MERGE_DENESTED_20260609.zip`

The denested merge of all code snapshots into a flat, navigable structure. "Denested" means the original deep directory nesting (e.g., `archive/archive/archive/file`) has been resolved to a single-level layout.

### Surface Anchor
`ORIGINS_SURFACE_ANCHOR_PULL_20260609.zip`

A NOEMA surface anchor — a stable snapshot of the entire OoL project as it existed on the NOEMA surface at the time of the pull (2026-06-09). This is the reference state against which future changes should be measured.

### Crystallized Search
`origins_NOEMA_PROJECT_SEARCH_CRYSTALLIZED_v03_to_v10_FULL/NOEMA_PROJECT_SEARCH_REPORT.json`

The crystallized output of NOEMA project search — a comprehensive index of all files found across the NOEMA surface related to the Origins project.

## 6.4 Canon Hierarchy

```
03_canon_solver/
├── ORIGINS_CODE_MERGE_DENESTED_20260609/     → Merged code (flat)
├── ORIGINS_SURFACE_ANCHOR_PULL_20260609/      → NOEMA surface reference
├── origins_FULL_CRYSTALLIZED_NOEMA_LINEAGE/    → Complete lineage
├── origins_LOCAL_NOEMA_ALL_FOUND_REPACK/       → All local traces
├── origins_NOEMA_PROJECT_SEARCH_CRYSTALLIZED/  → Search results
├── origins_RECONSTRUCTED_FROM_NOEMA_TRACE/     → Full reconstruction
└── origins_noema_internet_comparisons_v08/     → External validation
```

## 6.5 Relationship to 01_versions

| Aspect | 01_versions | 03_canon_solver |
|--------|-------------|-----------------|
| Purpose | Historical preservation | Clean reference |
| Granularity | One archive per version | Consolidated |
| Deduplication | None | Full |
| Nesting | Deep (archive/archive/) | Flat (denested) |
| Contains | All artifacts | Code + solver only |
| Authority | As-is snapshot | Curated reference |

## 6.6 Solver Core Identification

The true "solver" (the algorithmic engine) is distributed across:
1. **Kähler-Euler-Berry engine**: `OriginsKahlerEulerBerry.py` (present in every version archive)
2. **Main simulation**: `OriginOfLife.v*.py` (one per version)
3. **Phase bridge**: NOEMA phase semantic bridge (external, in runtime/)
4. **CIELingo pipeline**: linguistic-semantic processing (external, in canon/cielingo/)

The solver's mathematical core is the Kähler-Euler-Berry geometry — a ~500-line Python implementation that computes Berry phases on Kähler manifolds with Euler identity closure.
