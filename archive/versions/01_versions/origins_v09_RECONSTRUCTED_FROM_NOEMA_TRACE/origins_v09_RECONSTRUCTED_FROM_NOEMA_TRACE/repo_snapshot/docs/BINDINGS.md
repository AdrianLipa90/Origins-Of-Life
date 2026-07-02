# Bindings layer

This branch now contains two runtime-to-definition bridges:

- `src/origin_of_life/adapters.py` - transitional bridge added during the first integration pass
- `src/origin_of_life/bindings.py` - canonical cleaned bridge for future runtime adoption

## Canonical choice

Use `bindings.py` as the preferred integration layer.

It provides:

- `scenario_config_to_definition(...)`
- `scenario_config_to_entity_record(...)`
- `runtime_report_to_entity_record(...)`
- `build_runtime_catalog_snapshot(...)`
- `build_core_runtime_snapshot(...)`

## Why both exist

The branch was built incrementally and non-destructively.
To preserve auditability, the earlier bridge remains present, while `bindings.py` is the cleaned canonical layer.
