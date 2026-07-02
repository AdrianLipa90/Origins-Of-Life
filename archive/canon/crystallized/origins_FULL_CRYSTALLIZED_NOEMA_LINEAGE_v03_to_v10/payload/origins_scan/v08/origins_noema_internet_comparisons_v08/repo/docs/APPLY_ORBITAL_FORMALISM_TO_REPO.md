# Apply orbital formalism to `Origins-Of-Life`

This branch applies the orbital repository formalism as a non-destructive overlay.

## Source principles used from project files

The implementation follows the uploaded project files that define:

- semantic hierarchy: `relation -> identity -> memory -> process -> artifact`
- the need for full orbital identity of files/modules/processes
- the need to treat LLM as a temporary attractor rather than the center
- the idea that objects in the repo should have more than a path/hash and must expose orbit, phase, relation depth, and semantic mass
- the OrchORbital split:
  - orbital state space
  - orchestration / couplings
  - reduction / closure event

## What was added

### Code
- `origins/orbital_formalism.py`
- `scripts/export_orbital_manifest.py`

### Existing layer already present on `main`
- `origins/definitions.py`
- `origins/registry.py`
- `origins/catalog.py`
- `origins/bindings.py`

## Orbital interpretation of the repository

### Relation layer
The whole repository is treated as a relational system rather than a flat set of files.
The canonical hierarchy is:

`relation -> identity -> memory -> process -> artifact`

### Identity layer
Identity is expressed through `EntityRecord` and scenario/module canonical IDs.

### Memory layer
The repository manifest acts as explicit exported memory of the current orbital assignment.

### Process layer
The active process objects are:
- `origins/scenarios.py`
- `origins/simulator/universal.py`
- `origins/analysis/sweep.py`
- `scripts/run_simulation.py`
- `scripts/run_sweep.py`

### Artifact layer
Artifacts are exported as JSON manifests describing the orbital state of the repository.

## Current object mapping

### Scenario objects
Each scenario receives:
- `canonical_id`
- `orbit_index`
- `phase`
- `winding_number`
- `relation_depth`
- `semantic_mass`
- `subjective_time_scale`

### Module objects
The following are explicitly modeled as orbital nodes:
- scenarios module
- simulator module
- analysis module
- registry module
- definitions module
- run_simulation script
- run_sweep script

### Edges
The overlay currently creates explicit orbital edges such as:
- scenario -> scenarios module (`declared_in`)
- scenario -> simulator module (`executed_by`)
- scenario -> analysis module (`analyzed_by`)
- script -> runtime module (`launches`)
- simulator -> registry (`indexed_by`)
- simulator -> definitions (`constrained_by`)

## How to use

Export the current orbital manifest:

```bash
python scripts/export_orbital_manifest.py --out outputs/orbital_repository_manifest.json
```

Restrict export to selected scenarios:

```bash
python scripts/export_orbital_manifest.py --scenarios A E --out outputs/orbital_AE.json
```

## Current limit

This branch adds the formal orbital layer and export path, but does not yet overwrite the current runtime modules.
That was deliberate, to avoid damaging the working simulation pipeline.

## Next step

The next step is to integrate the orbital layer directly into:
- `origins/simulator/universal.py`
- `origins/analysis/sweep.py`
- `scripts/run_simulation.py`
- `scripts/run_sweep.py`

so that runtime outputs automatically emit orbital entity records and registry snapshots.
