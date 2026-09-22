# First-RNA Activator Context v0.10

Status: `PREREGISTERED STRUCTURAL CAPABILITY AUDIT / APPEND-ONLY`

Date: 2026-09-22  
Parent result: `v0.9 PASS_LOCAL_CONTEXT_REPRESENTATION_CALIBRATION`

## Question

Can the frozen v0.9 representation distinguish linkage regioselectivity under
different nucleotide activating groups?

## External observation

Huang & Ferris (2006), JACS 128, 8914-8919,
DOI `10.1021/ja061782k`, report that changing activation chemistry changes
the 3',5' linkage fraction.

Values quoted in that paper:

```
A:
  imidazole activation       ~0.67
  1-methyladenine activation  0.74

U:
  imidazole activation       ~0.20
  1-methyladenine activation  0.61
```

These values are evidence that activator identity can be an observable context
variable. They are not used to fit a v0.10 parameter.

## Frozen parent representation

v0.9 contains:

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

No v0.9 parameter or field may be added before this capability audit is run.

## Capability gate

The frozen representation must contain an explicit field/state that can
distinguish activation chemistry.

If not, verdict:

`FAIL_MISSING_ACTIVATOR_CONTEXT`

and numerical prediction/comparison is `NOT_RUN`.

A default assumption that activator identity has no effect is forbidden because
the external observation already shows condition dependence.

## Causal boundary

No v0.3 global rate, v0.5 extension multiplier, v0.9 local-order relation,
geometry, zeta, replication, or hidden activator coefficient may be changed.

## Interpretation

This is a structural capability audit, not predictive validation.

A FAIL means v0.9 improved local sequence/linkage representation but remains
unable to represent an experimentally important chemical-control variable.
