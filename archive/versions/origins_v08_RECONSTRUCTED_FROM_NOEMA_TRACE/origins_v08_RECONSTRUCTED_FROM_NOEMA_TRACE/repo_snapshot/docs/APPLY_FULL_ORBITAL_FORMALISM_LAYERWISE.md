# Full layerwise orbital formalism for `Origins-Of-Life`

This branch adds a dedicated `origins/orbital/` package that applies the repository formalism in layers.

## Layers added

### 1. Sphere
- `origins/orbital/sphere.py`
- `OrbitalSphere`
- `SphereEmbedding`

This layer models local Bloch-like charts, parent/child spheres and leak mode.

### 2. State
- `origins/orbital/state.py`
- `OrbitalCoordinate`
- `OrbitalSystemState`

This layer stores the per-entity orbital coordinate:
`(r, theta, phi, omega, tau_local, semantic_mass, attractor_charge, coherence, defect, seed_norm)`

### 3. Potentials
- `origins/orbital/potentials.py`
- `PotentialTerms`
- `compute_potential_terms(...)`

This layer provides:
- `V_EC`
- `V_ZS`
- `V_rel`
- `V_mem`
- `V_def`
- `V_ext`
- `V_tot`

### 4. Subjective time
- `origins/orbital/subjective_time.py`
- `compute_local_subjective_time(...)`

This layer implements the local time operator needed before winding can become non-decorative.

### 5. Winding
- `origins/orbital/winding.py`
- `WindingComponents`
- `compute_winding_components(...)`

The implementation uses component windings:
- EC
- ZS
- relational
- reduction

### 6. Memory
- `origins/orbital/memory.py`
- `ReductionResidue`
- `MemoryState`
- `apply_memory_update(...)`

This follows the project rule that memory is a residue of reduction, not its cause.

### 7. OORP
- `origins/orbital/oorp.py`
- `OORPTrace`
- `run_oorp_pipeline(...)`

This layer models:
- relation score
- orchestration score
- reduction score
- memory update

### 8. Repository assignment
- `origins/orbital/repository_assignment.py`
- `assign_orbital_state_to_entity(...)`
- `build_repository_system_state(...)`

This is the bridge from repo entities to orbital coordinates.

## Relation to the existing repo

The package is additive and non-destructive.
It does not overwrite the working simulator yet.
Instead it gives the repository a canonical orbital package that can now be wired into:
- `origins/simulator/universal.py`
- `origins/analysis/sweep.py`
- `scripts/run_simulation.py`
- `scripts/run_sweep.py`

## Branch purpose

This branch exists to carry the full orbital layers without risking the currently working runtime.
