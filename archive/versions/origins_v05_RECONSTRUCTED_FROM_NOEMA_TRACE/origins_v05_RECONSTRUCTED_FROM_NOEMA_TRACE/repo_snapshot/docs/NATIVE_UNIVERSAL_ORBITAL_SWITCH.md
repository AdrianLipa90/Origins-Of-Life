# Native logical switch for `universal.py`

This branch introduces a native orbital path for the simulator **without deleting or replacing** the original `origins/simulator/universal.py`.

## Added files

- `origins/orbital/bundle_builder.py`
- `origins/simulator/universal_orbital.py`
- `scripts/run_simulation_native_orbital.py`

## What this means

The original class remains:
- `origins/simulator/universal.py` -> `UniversalOriginSimulator`

The new native orbital class is:
- `origins/simulator/universal_orbital.py` -> `OrbitalUniversalOriginSimulator`

It subclasses the original runtime and adds:
- `build_entity_record()`
- `build_orbital_bundle()`
- `export_orbital_bundle()`
- extended `run(..., orbital_export=...)`
- extended `save_outputs(..., export_orbital=...)`

## Why this is the correct logical switch

This keeps the old path fully intact while making the orbital path native to the simulator package rather than purely external.

In other words:
- old path: `UniversalOriginSimulator`
- native orbital path: `OrbitalUniversalOriginSimulator`

## Entry point

```bash
python scripts/run_simulation_native_orbital.py --scenario A --hours 120 --outdir outputs_native_orbital
```

This produces standard outputs plus orbital bundle JSON.
