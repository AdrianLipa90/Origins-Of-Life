# Definitional runtime layer for `origins/`

This branch adds a definitional layer directly on top of the current `main` package layout.

## Purpose

The current repository already contains a strong executable structure:

- `origins/scenarios.py`
- `origins/simulator/universal.py`
- `origins/analysis/sweep.py`
- supporting subpackages for chemistry, biology, topology and cosmology

What was still missing was a canonical layer for:

- semantic hierarchy
- object identity
- epistemic status
- runtime-to-registry bindings
- catalog snapshots for scenarios and core modules

## Added files

- `origins/definitions.py`
- `origins/registry.py`
- `origins/catalog.py`
- `origins/bindings.py`

## Current mapping

### Relation layer
- `origins/definitions.py`

### Identity layer
- `origins/registry.py`
- `origins/catalog.py`

### Process layer
- `origins/scenarios.py`
- `origins/simulator/universal.py`
- `origins/analysis/sweep.py`

### Artifact layer
- runtime reports, CSVs, NPZ snapshots, heatmaps, exported manifests

## Next integration step

The next natural refactor is to wire the new layer into existing runtime files:

- `origins/simulator/universal.py`
- `origins/analysis/sweep.py`
- `scripts/run_simulation.py`
- `scripts/run_sweep.py`

That step should introduce:

- scenario entity records
- report entity records
- exported registry snapshots
- canonical IDs in summaries and manifests
