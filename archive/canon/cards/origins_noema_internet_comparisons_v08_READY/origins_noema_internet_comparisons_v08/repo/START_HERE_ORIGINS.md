# Start Here — Origins-Of-Life

This is the canonical onboarding entrypoint for the repository **as it exists today**.

## What this repository is

`Origins-Of-Life` is best understood as a computational abiogenesis / emergence repository.

Its current executable center is:
- origin-of-life scenarios
- universal emergence simulation
- topology / field modulation
- sweep-based comparison across habitats and parameters

## What this repository is *not* canonically centered on

The root `README.md` still contains legacy high-level framing around CIEL/0, CMB intelligence detection and cosmic signaling.
That material may remain historically relevant, but it is **not** the best semantic entrypoint for the current runnable repository surface.

## Canonical public semantic surface

Use:
- `origins.abiogenesis`
- `origins.abiogenesis.api`
- `scripts/run_abiogenesis.py`

Treat these as the repo-facing semantic layer.

## Internal implementation substrate

The lower-level execution substrate remains:
- `origins.orbital`
- `origins.simulator.universal_orbital`
- `origins.analysis.sweep_orbital`

Those names are acceptable internally, but public examples should prefer the `abiogenesis` surface.

## Recommended starting points

### Single scenario run
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

## Semantic canon

- public repo-facing semantics: `abiogenesis / emergence / habitat / feasibility / residue / recurrence`
- internal substrate semantics: `orbital`

If a future change introduces a second public semantic layer with equivalent meaning, that change should be rejected unless the canon is updated first.
