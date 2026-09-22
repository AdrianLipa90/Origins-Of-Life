# First-RNA Base-Resolved Reactive Chemistry v0.5

Status: `PREREGISTERED CALIBRATION / BASE IDENTITY ADDED / APPEND-ONLY`

Date: 2026-09-22  
Parent result: `v0.4 FAIL_BASE_SPECIFIC_HOLDOUT`

## Why this model extension is required

v0.4 reused the v0.3 A/U-calibrated activated-clay profile unchanged for the
held-out C condition and predicted a tail to the model ceiling (80 nt), while
the empirical C observation is about 20-25 nt after nine days.

The failure occurred with exact nucleotide conservation and no parameter
retuning. Therefore the missing variable is chemical identity, not numerical
stability.

## v0.5 state extension

The v0.3 conservative reactive-state dynamics are retained.

A nucleotide/base identity is added at the profile level:

```
A
U
C
G
```

No global v0.3 rate is changed.

The only new calibrated parameter in v0.5 is a base-specific effective
extension multiplier.

## Frozen calibration multipliers

```
A: 1.0
U: 1.0
C: 0.025
G: UNRESOLVED
```

A/U retain the v0.3 calibrated activated-clay extension rate exactly.

The C multiplier `0.025` is a coarse calibration choice made before the first
v0.5 CI execution so that the base-resolved model can represent the published
20-25 nt C tail after nine days.

It is not a measured kinetic constant and may not be cited as one.

G remains unresolved because Huang & Ferris report that G chain lengths could
not be reliably determined from the gel bands in that experiment.

## Frozen horizons

```
A: 24 h
U: 24 h
C: 216 h
G: no quantitative gate
```

## Frozen calibration gate

Using the same v0.3 tail metric (maximum length whose per-length nucleotide mass
fraction is >=1e-10):

A and U each require:

```
40 <= tail <= 55 nt
```

C requires:

```
20 <= tail <= 25 nt
```

All simulated bases require relative nucleotide conservation error <=1e-10.

G must remain explicitly unresolved and must not silently inherit an A/U/C
multiplier.

PASS:

`PASS_BASE_RESOLVED_CALIBRATION`

otherwise:

`FAIL_BASE_RESOLVED_CALIBRATION`

## Interpretation boundary

This is a repair/calibration after a held-out falsification. It is not an
independent prediction.

A PASS means the architecture now has enough state to encode the observed
base-dependent chain-length separation without changing the shared conservative
reaction skeleton.

## Next independent test

The next gate must not use chain length from the A/U/C calibration set.

Preferred held-out observables are linkage/regioselectivity or independent
adsorption/feeding-reaction data.
