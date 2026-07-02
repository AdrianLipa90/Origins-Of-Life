# Orbital runtime bridge

This branch now includes executable bridge entrypoints on top of the full orbital package.

## Added files

- `origins/orbital/runtime_bridge.py`
- `scripts/run_simulation_orbital.py`
- `scripts/run_sweep_orbital.py`

## Purpose

These files do not overwrite the existing simulator or analysis code.
Instead they wrap the current runtime and emit orbital outputs.

## Single scenario run

```bash
python scripts/run_simulation_orbital.py --scenario A --hours 120 --outdir outputs_orbital
```

This emits:
- normal simulation outputs
- orbital bundle JSON containing:
  - entity record
  - orbital coordinate
  - potential terms
  - winding
  - OORP trace
  - memory state
  - output paths

## Sweep run

```bash
python scripts/run_sweep_orbital.py --scenarios A E --hours 60 --outdir outputs_orbital_sweep
```

This emits:
- standard sweep outputs
- `orbital_system_state.json`
- `orbital_repository_snapshot.json`

## Why bridge instead of overwrite

The repository already has a working runtime. The bridge allows immediate use of the orbital package without risking regression in the core simulator.
