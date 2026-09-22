# First-RNA Local Linkage Context v0.9 — Result

Status: `PASS_LOCAL_CONTEXT_REPRESENTATION_CALIBRATION`

Date: 2026-09-22  
Execution head: `b1e4502d15ed5d78ac4eb8965da2735d715b9bb4`

## What was added

v0.9 adds explicit symbolic local context without assigning numerical kinetic
weights:

```
DimerContext:
  left_class   in {Pu, Py}
  linkage      in {3p5, 2p5}
  right_class  in {Pu, Py}

GrowthContext:
  terminal_class    in {Pu, Py}
  previous_linkage  in {3p5, 2p5}
  incoming_class    in {Pu, Py}
```

There is no chain-length field in this minimal local-context state.

## Frozen categorical calibration

Source: Miyakawa & Ferris (2003), DOI `10.1021/ja034328e`.

The source ordering is represented exactly as an ordinal relation:

```
{Pu-3p5-Py}
>
{Pu-3p5-Pu, Pu-2p5-Py}
>
{Pu-2p5-Pu}
```

The equality class and both strict inequalities are preserved.

No numeric score, probability, energy, rate, or pair-specific correction was
introduced.

## Gate result

```
distinct_source_contexts = true
source_equality_preserved = true
source_strict_inequalities_preserved = true
growth_context_complete = true
chain_length_term_present = false
numeric_context_parameter_added = false
```

Verdict:

`PASS_LOCAL_CONTEXT_REPRESENTATION_CALIBRATION`

## Epistemic boundary

This is a representation/calibration PASS, not predictive validation.

A 2006 study reports that structural product proportions for oligo(A) and
oligo(U) did not vary with chain length over the analyzed fractions. That
observation had already been inspected before v0.9 was preregistered, so it is
recorded only as external consistency, not as a held-out prediction.

## Causal boundary

```
global_rates_retuned = false
base_extension_multipliers_retuned = false
numeric_context_parameter_added = false
geometry_used = false
zeta_used = false
replication_used = false
predictive_validation_claim = false
```

## CI / receipt

Dedicated workflow:

- run: `35791829389`
- conclusion: `success`
- dedicated tests: `4 passed in 0.35s`
- artifact id: `10722251820`
- artifact digest:
  `sha256:2a12c6feb2014b9bbf52d038bca1f82c930ec4c0ba43b4f0d70869a2a669ed3a`

Abiogenesis Canon:

- run: `35791829271`
- conclusion: `success`
- repository-wide tests: `127 passed in 3.72s`

## Interpretation

v0.8 showed that a universal additive growth-stage correction is insufficient.
v0.9 now has enough symbolic state to express the experimentally reported local
sequence/regioselectivity structure without inventing a new fitted constant.

The next gate should not reward this representation for reproducing the same
ordering used to define it. It must use a new observable or condition.

A particularly strong next target is activation chemistry: published data show
that changing the activating group can alter 3',5' linkage fractions strongly,
especially for U. That should be tested as a separate context variable rather
than folded into a free pair-specific coefficient.
