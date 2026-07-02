# Abiogenesis public surface for `Origins-Of-Life`

This repository now distinguishes two layers:

- `origins.orbital` — implementation substrate
- `origins.abiogenesis` — canonical public semantic surface

## Public imports

Prefer:

```python
from origins.abiogenesis import (
    OriginHabitatShell,
    EmergenceCoordinate,
    FeasibilityTerms,
    HistoricalMemory,
    AbiogenesisRuntimeAdapter,
)
```

or:

```python
from origins.abiogenesis.api import (
    create_abiogenesis_runtime,
    run_habitat_scan,
    run_feasibility_scan,
    run_origin_comparison,
)
```

## Canonical runner

```bash
python scripts/run_abiogenesis.py --mode single --scenario A --hours 120 --outdir outputs_abiogenesis
```

## Policy

- Repo-facing docs and examples should prefer `abiogenesis` names.
- `orbital` names remain valid internally as the execution substrate.
- Do not introduce a second public semantic surface with equivalent meaning.
