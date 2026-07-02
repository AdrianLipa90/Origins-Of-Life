# Systematized origin-of-life runtime

This branch introduces a non-destructive modular layout under `src/origin_of_life/`.

## New package layout

- `utils.py` - filesystem helpers
- `plotting.py` - figure saving helper
- `grid.py` - Laplacian backend with optional numba acceleration
- `scenarios.py` - canonical `ScenarioConfig` and the five scenario presets
- `sweep_v3.py` - legacy prebiotic sweep extracted from the monolith
- `simulator.py` - `UniversalOriginSimulator` extracted as the runtime core
- `sweeps.py` - topology x synthesis parameter sweeps
- `runners.py` - scenario batch runners and quick tests
- `cli.py` - new CLI entrypoint with real argument parsing

## Important note

The original monolithic files remain in place on this branch for safety and auditability.
This branch adds a structured runtime without destructive file moves.
