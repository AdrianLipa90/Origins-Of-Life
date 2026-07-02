# Registry layer

The project blueprint requires a topological identity layer for important system entities.

## Minimal record

`EntityRecord` contains:

- `canonical_id`
- `object_type`
- `source_path`
- `sector`
- `orbit_index`
- `phase`
- `winding_number`
- `relation_depth`
- `epistemic_status`
- `provenance_links`
- `dependency_links`

## Extended record used on this branch

The project notes also motivate extending the record with:

- `semantic_mass`
- `subjective_time_scale`
- `sphere_id`
- `parent_sphere_id`
- `attractor_weights`
- `leak_mode`

## Current branch status

The runtime is not yet fully driven by `EntityRecord`.
However, the canonical schema now exists in:

- `src/origin_of_life/registry.py`
- `src/origin_of_life/definitions.py`

This prepares the next step: attaching real identity and orbital metadata to runtime entities such as scenarios, simulator outputs and reports.
