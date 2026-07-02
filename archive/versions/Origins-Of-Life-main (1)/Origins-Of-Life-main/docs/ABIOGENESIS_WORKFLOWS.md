# Abiogenesis workflow presets

These presets are the recommended repo-facing workflows on top of the canonical `origins.abiogenesis` surface.

They are not alternative public semantic layers.
They are opinionated usage presets.

## Module

```python
from origins.abiogenesis.workflows import (
    default_single_origin_protocol,
    default_habitat_protocol,
    default_feasibility_protocol,
    default_origin_comparison_protocol,
)
```

## Recommended uses

### 1. Single-origin protocol
Use when you want a single scenario run with standard orbital exports.

### 2. Habitat protocol
Use when you want the sweep-v3 habitat landscape.

### 3. Feasibility protocol
Use when you want topology/parameter feasibility scanning.

### 4. Origin comparison protocol
Use when you want a cross-scenario comparison under the canonical repo-facing interface.

## Why these presets exist

The repository already contains several low-level execution paths.
The presets reduce surface complexity while preserving one semantic canon.
