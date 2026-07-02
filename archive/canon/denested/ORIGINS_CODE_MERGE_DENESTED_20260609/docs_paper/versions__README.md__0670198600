# Origins-Of-Life

`Origins-Of-Life` is a computational abiogenesis / emergence repository.

The current runnable surface of the project is centered on:
- origin-of-life scenarios
- universal emergence simulation
- topology / field modulation
- habitat and feasibility sweeps
- cross-scenario comparison

## Canonical public surface

Use the repository through the repo-facing semantic layer:
- `origins.abiogenesis`
- `origins.abiogenesis.api`
- `scripts/run_abiogenesis.py`

The lower-level orbital machinery remains available as an implementation substrate under:
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

## Historical note

The repository still contains legacy theoretical framing around CIEL/0, CMB intelligence detection and broader cosmic signaling concepts. That material remains historically relevant to the project’s evolution, but it is not the best semantic entrypoint for the current runnable repository surface.
