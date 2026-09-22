# First-RNA Pair-Context Holdout v0.8

Status: `PREREGISTERED CROSS-CONDITION TRANSFER / APPEND-ONLY`

Date: 2026-09-22  
Parent result: `v0.7 FAIL_LENGTH_INVARIANT_LINKAGE_POINT_ESTIMATE`

## Question

Is the increase in 3',5' linkage fraction from dimer to trimer a universal
growth-stage shift that transfers unchanged across binary monomer pairs?

## Calibration condition: A/U

Ertem et al. (2007), Astrobiology 7, 715-722,
DOI `10.1089/ast.2007.0138`:

```
A/U dimer 3',5' fraction  = 0.51
A/U trimer 3',5' fraction = 0.61
```

Frozen stage shift:

```
delta_stage = 0.61 - 0.51 = +0.10
```

## Held-out transfer condition: A/C

Ertem et al. (2008), International Journal of Astrobiology 7(1), 1-7,
DOI `10.1017/S147355040700393X`:

```
A/C dimer 3',5' fraction  = 0.49
A/C trimer 3',5' fraction = 0.56
```

The held-out trimer value MUST NOT be used to set the prediction.

Frozen transferred prediction:

```
p35_AC_trimer(predicted) = 0.49 + 0.10 = 0.59
```

## Rounding-only gate

The reported percentages are integer percentages. No measurement-error model is
invented.

A conservative envelope is propagated from rounding alone:

```
A/U dimer 51%  -> [0.505, 0.515]
A/U trimer 61% -> [0.605, 0.615]
delta_stage envelope -> [0.090, 0.110]

A/C dimer 49%  -> [0.485, 0.495]
predicted A/C trimer envelope -> [0.575, 0.605]

observed A/C trimer 56% -> [0.555, 0.565]
```

If the transferred prediction envelope and observed A/C envelope are disjoint:

`FAIL_UNIVERSAL_STAGE_SHIFT`

Otherwise:

`PASS_UNIVERSAL_STAGE_SHIFT_ROUNDING_COMPATIBILITY`

This is not a statistical-significance test.

## Interpretation boundary

A FAIL falsifies the exact universal-additive stage-shift model. It does not
prove a unique mechanism. Plausible missing context includes monomer-pair /
local sequence identity, previous-linkage context, or another condition that
differs across the two binary systems.

No v0.3 kinetic rate, v0.5 extension multiplier, geometry, zeta, replication,
or free correction parameter may be introduced in v0.8.
