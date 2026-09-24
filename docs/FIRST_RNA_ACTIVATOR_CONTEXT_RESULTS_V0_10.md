# First-RNA Activator Context v0.10 — Result

Status: `FAIL_MISSING_ACTIVATOR_CONTEXT`

Date: 2026-09-22  
Execution head: `f6a06639ba570291a163898fc5e2aa33fbea37f0`

## External observation

Huang & Ferris (2006), DOI `10.1021/ja061782k`, report strong dependence of
3',5' linkage fraction on activation chemistry.

Quoted values:

```
A:
  imidazole          ~0.67
  1-methyladenine     0.74

U:
  imidazole          ~0.20
  1-methyladenine     0.61
```

## Capability audit

The frozen v0.9 representation contains:

```
DimerContext:
  left_class
  linkage
  right_class

GrowthContext:
  terminal_class
  previous_linkage
  incoming_class
```

No field or state distinguishes activation chemistry.

Therefore:

```
explicit_activator_context = false
numerical_comparison = NOT_RUN
verdict = FAIL_MISSING_ACTIVATOR_CONTEXT
```

No default "activator-independent" prediction was manufactured.

## Interpretation

v0.9 repaired local base/linkage context but remains structurally unable to
represent a chemical-control variable with a documented large effect on
regioselectivity.

The particularly large U shift (about 0.20 to 0.61 in the cited comparison)
makes it inappropriate to hide activation chemistry inside a generic
pair-specific correction.

The next admissible model extension is an explicit activator/activation-state
coordinate. Any numerical calibration must be separated from later validation.

## Causal boundary

```
activator_parameter_added = false
global_rates_retuned = false
base_extension_multipliers_retuned = false
local_context_relation_retuned = false
geometry_used = false
zeta_used = false
replication_used = false
predictive_validation_claim = false
```

## CI / receipt

Dedicated workflow:

- run: `35792160794`
- conclusion: `success`
- dedicated tests: `4 passed in 0.35s`
- artifact id: `10722542449`
- artifact digest:
  `sha256:943d7f77bbaeb8b6dd2311f52840a8c0e405dfb1dba4a988212d775c53b2f1db`

Abiogenesis Canon:

- run: `35792160637`
- conclusion: `success`
- repository-wide tests: `131 passed in 2.87s`

## Next gate

Add explicit activation chemistry as state, but do not call a fit to the
already-inspected Huang-Ferris values a prediction.

A subsequent predictive gate requires a separate activation condition or
independent experiment not used to choose the activator-state calibration.
