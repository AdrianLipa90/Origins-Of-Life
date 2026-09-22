# Zeta Spectral Specificity Results v0.2

Status: `FAIL_ZERO_SPECIFICITY / PREREGISTERED NEGATIVE RESULT / APPEND-ONLY`

Date: 2026-09-22  
Preregistration commit: `7999001817d4fe461fda79e100ed803a4d340bd9`  
Execution head: `a8f4501810f0fc038367dd71a549520b8bbaecd6`  
Canonical test gate: `93 passed`

## Frozen design

The experiment followed `ZETA_SPECTRAL_SPECIFICITY_PREREG_V0_2.md` without
changing seeds, endpoints, mask definitions or verdict thresholds after endpoint
inspection.

- scenarios: A and C
- seeds: 201--220 inclusive
- 20 matched seeds per scenario
- 120 h
- 32x32 grid
- dt = 0.05 h
- geometry enabled
- clay enabled
- pre-seeded RNA spatial model
- seven arms:
  - none
  - noise_only
  - narrow_zeta
  - narrow_reflected
  - narrow_shifted
  - narrow_equispaced
  - histogram_permuted

For every seed, all seven arms had one identical pre-operator initial-state
digest. No matched-state violation occurred.

## Strongest placebo validation

The `histogram_permuted` mask preserved the exact sorted non-DC mask-value
multiset of `narrow_zeta`:

```
narrow_zeta sorted mask SHA256:
ab1063745ca4a2db036a3ab3aab166a6bdb36031b72bf6b74fd90e79387d853b

histogram_permuted sorted mask SHA256:
ab1063745ca4a2db036a3ab3aab166a6bdb36031b72bf6b74fd90e79387d853b
```

Both have mean non-DC transmission `0.9846603450798519`, while their spatial
mask correlation is approximately `-0.02036`. The placebo therefore keeps the
attenuation histogram exactly while moving attenuation to different Fourier-bin
conjugacy classes.

## Primary endpoint

Primary endpoint: connected protocell-component count at 120 h.

Exact two-sided paired sign-flip tests were computed on the 20 seedwise
differences `narrow_zeta - placebo`. Holm correction was applied jointly over
all eight scenario x placebo tests.

| Scenario | Placebo | Mean Δ | Median Δ | Median |Δ| | + / - / 0 | raw p | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|
| A | reflected | -0.50 | -1.0 | 3.0 | 7 / 11 / 2 | 0.569084 | 1.000 |
| A | shifted | 0.00 | -1.0 | 2.0 | 7 / 11 / 2 | 1.000000 | 1.000 |
| A | equispaced | -0.05 | 0.0 | 1.0 | 8 / 8 / 4 | 1.000000 | 1.000 |
| A | histogram_permuted | -0.50 | -1.0 | 1.5 | 7 / 11 / 2 | 0.407639 | 1.000 |
| C | reflected | +0.25 | 0.0 | 0.0 | 5 / 0 / 15 | 0.062500 | 0.500 |
| C | shifted | +0.25 | 0.0 | 0.0 | 6 / 1 / 13 | 0.125000 | 0.875 |
| C | equispaced | +0.10 | 0.0 | 0.0 | 3 / 1 / 16 | 0.625000 | 1.000 |
| C | histogram_permuted | +0.15 | 0.0 | 0.0 | 5 / 2 / 13 | 0.453125 | 1.000 |

No adjusted p-value reaches 0.05.

For the exact-histogram placebo, the mean effect also changes sign between the
two scenarios:

```
Scenario A: -0.50 components
Scenario C: +0.15 components
```

This independently violates the preregistered cross-scenario sign-consistency
condition.

## Full paired vectors

### Scenario A

```
reflected:
[3, 3, 3, -3, -1, 2, 0, -8, -5, -4, -2, -3, -1, 2, -1, 0, -1, 4, -4, 6]

shifted:
[2, 4, -1, -1, -3, 4, 0, -2, -3, -6, -2, -3, -2, 6, -1, 0, 3, 2, -1, 4]

equispaced:
[1, 3, 3, -1, 0, 0, -1, 1, -5, -2, -1, -1, -2, 2, 1, 1, 0, 0, -1, 1]

histogram_permuted:
[2, 4, 3, -3, -1, -1, -1, -3, 1, -4, 0, 0, -3, 3, -1, -3, -1, 1, -4, 1]
```

### Scenario C

```
reflected:
[0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]

shifted:
[0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 1, 1, 0]

equispaced:
[0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]

histogram_permuted:
[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, -1, 1, 0, 0, 0, 0, -1, 0, 1, 0]
```

## Endpoint means by arm

### Scenario A

| Mode | Protocells | Protocell area | Polymer-threshold px | mean_R | max_R |
|---|---:|---:|---:|---:|---:|
| none | 20.80 | 79.45 | 89.75 | 0.014110 | 0.056001 |
| noise_only | 36.60 | 85.10 | 113.35 | 0.014135 | 0.069679 |
| narrow_zeta | 35.95 | 84.05 | 113.00 | 0.014135 | 0.068707 |
| narrow_reflected | 36.45 | 84.15 | 112.10 | 0.014129 | 0.067997 |
| narrow_shifted | 35.95 | 83.30 | 111.70 | 0.014133 | 0.067960 |
| narrow_equispaced | 36.00 | 84.15 | 112.30 | 0.014135 | 0.068778 |
| histogram_permuted | 36.45 | 83.20 | 111.05 | 0.014130 | 0.067811 |

### Scenario C

| Mode | Protocells | Protocell area | Polymer-threshold px | mean_R | max_R |
|---|---:|---:|---:|---:|---:|
| none | 4.75 | 5.95 | 973.90 | 0.051536 | 0.241469 |
| noise_only | 3.45 | 4.05 | 652.05 | 0.057629 | 0.393774 |
| narrow_zeta | 3.55 | 4.25 | 661.05 | 0.057848 | 0.370306 |
| narrow_reflected | 3.30 | 3.85 | 660.55 | 0.057821 | 0.366410 |
| narrow_shifted | 3.30 | 3.80 | 658.90 | 0.057798 | 0.370650 |
| narrow_equispaced | 3.45 | 4.15 | 660.55 | 0.057780 | 0.378916 |
| histogram_permuted | 3.40 | 4.00 | 656.90 | 0.057771 | 0.366242 |

## Preregistered verdict

No placebo family satisfies the preregistered conditions in both scenarios.

```
reflected:            FAIL
shifted:              FAIL
equispaced:           FAIL
histogram_permuted:   FAIL

families passing: 0 / 4
verdict: FAIL_ZERO_SPECIFICITY
```

The `PARTIAL_ZERO_SPECIFICITY` threshold also fails because it required at
least two independent placebo families to pass in both scenarios.

## Scientific interpretation

The software evidence now supports a stronger negative statement than v0.1:

> Within the current Origins-Of-Life simulator, the tested biological endpoint
> changes do not depend reproducibly on the specific placement of the mapped
> first-six Riemann-zeta ordinates.

The legacy broad operator can still produce large changes, but v0.1 showed that
those changes are attributable to broad spectral smoothing/redistribution rather
than the detailed zeta-zero spacing.

Therefore the zeta layer must remain quarantined as an experimental/legacy
regularizer. It cannot currently be used as evidence for a zeta-specific
abiogenesis mechanism.

This is not a mathematical statement about the Riemann zeta function and not a
physical no-go theorem.

## Next action

Do not spend further model degrees of freedom tuning the zeta target map against
these endpoints.

The causal program should now return to the polymer-free / first-RNA candidate
chain, where the open question is chemically and biologically meaningful:
whether monomer -> oligomer -> replicator-like transition can be made
conservative, non-preseeded, reproducible and externally bindable.
