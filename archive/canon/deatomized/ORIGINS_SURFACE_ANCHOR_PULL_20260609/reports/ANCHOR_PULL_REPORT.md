# ORIGINS Surface Anchor Pull — local artifact reconstruction
Epistemic status: **artifact-grounded local reconstruction**, not external NOEMA SoT/CURRENT.
Created UTC: `2026-06-09T09:21:46Z`
## Counts
- Input files: 20
- Indexed file records: 4338
- Unique SHA-256 blobs: 275
- Normalized paths: 2738
- Anchor records: 638
- Code records: 2442
- Paper/docs records: 588
- Selected pull files: 2526
## Source stats
| source | files | unique_sha | anchors | code | bytes |
|---|---:|---:|---:|---:|---:|
| `Origins-Of-Life-claude-explore-repository-SsBL1.zip` | 52 | 48 | 0 | 38 | 640728 |
| `Origins-Of-Life-main (1).zip` | 121 | 116 | 1 | 90 | 802161 |
| `Origins-Of-Life-main.zip` | 22 | 18 | 0 | 10 | 533868 |
| `originsHolo2.py` | 1 | 1 | 0 | 1 | 37411 |
| `origins_FULL_CRYSTALLIZED_NOEMA_LINEAGE_v03_to_v10.zip` | 286 | 203 | 75 | 108 | 2364111 |
| `origins_LOCAL_NOEMA_ALL_FOUND_REPACK_v03_to_v10.zip` | 286 | 203 | 75 | 108 | 3319241 |
| `origins_NOEMA_PROJECT_SEARCH_CRYSTALLIZED_v03_to_v10_FULL.zip` | 850 | 203 | 222 | 324 | 6125766 |
| `origins_RECONSTRUCTED_FROM_NOEMA_TRACE_v03_to_v10_FULL.zip` | 1423 | 224 | 107 | 975 | 12985800 |
| `origins_dna_four_blocks_v05_RESULTS_ONLY.zip` | 9 | 9 | 5 | 0 | 25185 |
| `origins_noema_dna_to_protocell_v06_RESULTS_ONLY.zip` | 6 | 6 | 2 | 0 | 9863 |
| `origins_noema_internet_comparisons_v08_READY.zip` | 228 | 189 | 53 | 108 | 1547702 |
| `origins_non_arbitrary_pass_without_closure_v10_RESULTS_ONLY.zip` | 54 | 39 | 21 | 0 | 204586 |
| `origins_planetary_biology_app.zip` | 50 | 50 | 0 | 29 | 352203 |
| `origins_v03_RECONSTRUCTED_FROM_NOEMA_TRACE.zip` | 156 | 149 | 6 | 109 | 961393 |
| `origins_v05_RECONSTRUCTED_FROM_NOEMA_TRACE.zip` | 153 | 144 | 8 | 109 | 959553 |
| `origins_v05b_RECONSTRUCTED_FROM_NOEMA_TRACE.zip` | 161 | 151 | 18 | 107 | 955159 |
| `origins_v06_RECONSTRUCTED_FROM_NOEMA_TRACE.zip` | 160 | 149 | 15 | 109 | 973451 |
| `origins_v07_RECONSTRUCTED_FROM_NOEMA_TRACE.zip` | 162 | 149 | 16 | 109 | 1207234 |
| `origins_v08_RECONSTRUCTED_FROM_NOEMA_TRACE.zip` | 157 | 147 | 14 | 107 | 1009078 |
| `originsphaseholoext (1).py` | 1 | 1 | 0 | 1 | 74465 |

## What was pulled
Pulled files are grouped under `selected_files/anchors`, `selected_files/code`, `selected_files/paper_docs`, and `selected_files/reports_results`. Selection prioritized CURRENT/index/manifest/SHA/reconstruction anchors, Python/code, paper LaTeX/BibTeX/Markdown, and result reports/tables.

## Key anchor strategy
- Normalize archive-specific roots such as `origins_vXX_RECONSTRUCTED_FROM_NOEMA_TRACE/`, `repo_snapshot/`, `payload/`, and search-scan wrappers.
- Deduplicate by SHA-256 but keep up to 3 divergent variants per normalized path when version history differs.
- Avoid treating nested source ZIPs as proof of external surface access; nested ZIPs are indexed as files, but not recursively unpacked in this pass.

## Review files
- `reports/all_files_index.csv` — full index.
- `reports/selected_files_manifest.csv/json` — selected pull manifest.
- `reports/normalized_path_presence_top500.json` — repeated/divergent normalized paths.
- `reports/ANCHOR_PULL_SUMMARY.json` — machine summary.
