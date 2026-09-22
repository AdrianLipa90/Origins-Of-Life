# First-RNA Explicit Linkage State v0.7 — Result

Status: `FAIL_LENGTH_INVARIANT_LINKAGE_POINT_ESTIMATE`

Date: 2026-09-22  
Execution head: `b14e709a4dbd258633371571a56f70cb12ed7111`

## Calibration and held-out check

Source: Ertem, Hazen & Dworkin (2007), Astrobiology 7, 715-722,
DOI `10.1089/ast.2007.0138`.

Frozen dimer calibration:

```
p35(dimer) = 0.51
```

Frozen zero-order hypothesis:

```
p35(L) = 0.51
```

Therefore the preregistered trimer prediction was:

```
p35(trimer, predicted) = 0.51
```

Reported trimer point estimate:

```
p35(trimer, observed) = 0.61
```

Difference:

```
predicted - observed = -0.10
```

## Rounding-only compatibility

No experimental uncertainty model was invented.

Using only integer-percentage rounding:

```
51% -> [0.505, 0.515)
61% -> [0.605, 0.615)
```

The intervals are disjoint.

Therefore the length-invariant linkage point-estimate model is incompatible
with the reported rounded values.

This is not a statistical significance claim.

## Representation result

v0.7 successfully adds the previously missing explicit linkage state:

```
B_35
B_25
```

with normalized expected bond weight:

```
B_35 + B_25 = 1
```

So the v0.6 representation failure is repaired, but the first zero-order
dynamical hypothesis over that state is falsified.

## Causal boundary

```
global_rates_retuned = false
base_extension_multipliers_retuned = false
sequence_pair_term_added = false
chain_length_linkage_term_added = false
geometry_used = false
zeta_used = false
replication_used = false
statistical_significance_claim = false
```

## CI / receipt

Dedicated workflow:

- run: `35790517185`
- conclusion: `success`
- dedicated tests: `4 passed in 0.43s`
- artifact id: `10722031123`
- artifact digest:
  `sha256:d6028524cb986cbb998dc07ffbf71ac0ad12b010ea2bbdca8a132ae8c3217dda`

Abiogenesis Canon:

- run: `35790517065`
- conclusion: `success`
- repository-wide tests: `119 passed in 3.31s`

## Interpretation

The next missing variable is no longer "linkage identity" itself. The data now
require linkage selectivity to depend on additional context.

Admissible next candidates are:

1. chain-length / growth-stage dependence,
2. base-pair or local sequence context,
3. a coupled term containing both.

The next version must distinguish these rather than adding a free correction
term that simply reproduces 0.61.
