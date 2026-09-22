# Origins-Of-Life

`Origins-Of-Life` is a computational abiogenesis / emergence repository.

The current runnable surface of the project is centered on:
- origin-of-life scenarios
- universal emergence simulation
- topology / field modulation
- habitat and feasibility sweeps
- cross-scenario comparison

## Current model boundary

The default `UniversalOriginSimulator` uses a **pre-seeded RNA population** as
an explicit initial condition. It therefore models chemical-field evolution,
RNA-like population dynamics and protocell formation after polymers already
exist; it is not by itself a monomer-to-first-RNA proof.

For a polymer-free spatial start, use `preseed_rna=False`. The separate
`origins.biology.first_rna` module contains the current monomer/oligomer
candidate path.

The topology/Bloch/geometric-phase and zeta-indexed layers are experimental
model operators with open physical binding. They must be evaluated by ablation
against chemistry-only and matched-control runs.

## Exotic biology status

World environment and biology hypothesis are now separate objects. The active
water/RNA-like runtime is the `WATER_REFERENCE` baseline. Scenario C is bound
to `AMMONIA_CANDIDATE` and scenario D to `HYDROCARBON_CANDIDATE`.

The universal simulator remains a terracentric control for both exotic
profiles and labels those outputs `TERRACENTRIC_CONTROL_ONLY`.

`AMMONIA_CANDIDATE` now also has a separate
`AMMONIA_CANDIDATE_V0_1` computational candidate runtime. It is non-RNA and
non-lipid, conserves its modeled material sector, and operationalizes the five
relational life invariants. Its molecular/physical binding remains `OPEN` and
its default coefficients are unvalidated dimensionless candidate parameters.

`HYDROCARBON_CANDIDATE` now has a separate
`HYDROCARBON_CANDIDATE_V0_1` computational candidate runtime with an explicit
interfacial/aggregate reservoir. It makes no specific azotosome claim and keeps
all molecular binding OPEN.

The common candidate-life contract is relational rather than molecular:
bounded system, energy throughput, persistent information state, heritable
state transformation, and selection/differential persistence.

See `docs/EXOTIC_BIOLOGY_FRAMEWORK_V0_1.md` and
`docs/AMMONIA_CANDIDATE_RUNTIME_V0_1.md` and
`docs/HYDROCARBON_CANDIDATE_RUNTIME_V0_1.md`.
Matched cross-runtime ablations are defined in `docs/EXOTIC_RELATIONAL_BENCHMARK_V0_1.md`; they compare shared relational observables only and explicitly forbid chemistry ranking.


## Canonical public surface

Use the repository through the repo-facing semantic layer:
- `origins.abiogenesis`
- `origins.abiogenesis.api`
- `scripts/run_abiogenesis.py`

The lower-level orbital machinery remains available as an experimental implementation substrate under:
- `origins.orbital`
- `origins.simulator.universal_orbital`
- `origins.analysis.sweep_orbital`

For repo-facing usage, prefer the `abiogenesis` surface.

## Quick start

### Single scenario
```bash
python scripts/run_abiogenesis.py --mode single --scenario A --hours 120 --outdir outputs_abiogenesis
```

### Habitat scan
```bash
python scripts/run_abiogenesis.py --mode habitat_scan --outdir outputs_abiogenesis
```

### Feasibility scan
```bash
python scripts/run_abiogenesis.py --mode feasibility_scan --scenarios A E --hours 60 --outdir outputs_abiogenesis
```

### Cross-scenario comparison
```bash
python scripts/run_abiogenesis.py --mode origin_comparison --hours 120 --outdir outputs_abiogenesis
```

## Python entrypoints

### Public semantic API
```python
from origins.abiogenesis import (
    OriginHabitatShell,
    EmergenceCoordinate,
    FeasibilityTerms,
    HistoricalMemory,
    AbiogenesisRuntimeAdapter,
)
```

### High-level repo-facing workflows
```python
from origins.abiogenesis.api import (
    create_abiogenesis_runtime,
    run_habitat_scan,
    run_feasibility_scan,
    run_origin_comparison,
)
```

## Repository structure

### Public repo-facing layer
- `origins/abiogenesis/`
- `scripts/run_abiogenesis.py`
- `START_HERE_ORIGINS.md`

### Internal execution substrate
- `origins/orbital/`
- `origins/simulator/`
- `origins/analysis/`

## Semantic canon

For this repository:
- public repo-facing semantics = `abiogenesis / emergence / habitat / feasibility / residue / recurrence`
- internal implementation substrate = `orbital`

This means the project should not introduce a second equal-status public semantic layer for the same concepts.

## Onboarding and docs

Start here:
- `START_HERE_ORIGINS.md`

Additional docs:
- `docs/ABIOGENESIS_PUBLIC_SURFACE.md`
- `docs/ABIOGENESIS_WORKFLOWS.md`
- `docs/REPO_POSITIONING_AND_ADOPTION.md`

## Archive (extracted 2026-07-02)

The `archive/` directory contains the full project extraction from the master bundle `ORIGINS_OF_LIFE_FULL_ZIP_ONLY_SET_20260702.zip`:

| Directory | Source | Description | Files |
|-----------|--------|-------------|-------|
| `archive/versions/` | v03–v10 NOEMA trace reconstructions | Historical snapshots (13 archives) | 2714 |
| `archive/results/` | RESULTS_ONLY v05+v06 | Simulation outputs and metrics | 30 |
| `archive/canon/` | Full canon solver (denested, crystallized, reconstructed, atomized) | Code merges, traces, search results | 5138 |
| `archive/latex/` | TeX/Bib/Class from all archives | Paper sources and styles | 160 |
| `archive/failures/` | v10 non-arbitrary pass without closure | Semantic closure failure record | 54 |
| `archive/docs/` | 9-chapter documentation | Theory, computation, chronology, analysis, closure | 10 |

## Historical note

The repository still contains legacy theoretical framing around CIEL/0, CMB intelligence detection and broader cosmic signaling concepts. That material remains historically relevant to the project’s evolution, but it is not the best semantic entrypoint for the current runnable repository surface.
