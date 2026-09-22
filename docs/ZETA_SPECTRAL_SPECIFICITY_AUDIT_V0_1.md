# Zeta Spectral Specificity Audit v0.1

Status: `FAIL / SPECIFICITY QUARANTINE / APPEND-ONLY`

Date: 2026-09-22  
Branch: `experiments/matched-factorial-ablation-v01`

## Scope

This audit asks a narrower question than the matched abiogenesis ablation:

> Does the implemented spectral mask depend materially on the individual
> mapped Riemann-zero ordinates, or does it behave like a generic broad-band
> smoothing operator?

It does not test whether zeta zeros have a physical role in abiogenesis.

## Current mapping

The first six ordinates are mapped by

```
target(gamma) = 0.5 * gamma / (1 + gamma)
```

to approximately:

```
0.46696334
0.47729543
0.48077729
0.48408905
0.48526599
0.48704200
```

The full target span is only about `0.02008`.

## Width audit

The current notch factor is

```
1 - exp(-lambda_soft * (k - target)^2)
```

For one notch, half transmission occurs at

```
|k-target| = sqrt(ln(2) / lambda_soft)
```

For Scenario A, `lambda_soft=6`, giving a half-power width of about
`0.3399` cycles/sample. This is much wider than the entire six-target span.

The product of six such broad factors therefore suppresses most non-DC spatial
frequencies rather than isolating six narrow spectral neighborhoods.

At `32x32`, `lambda_soft=6`, the observed mask diagnostics are approximately:

- mean non-DC transmission: `3.48e-3`
- median non-DC transmission: `5.95e-8`
- fraction of non-DC coefficients below `1e-3`: `~0.828`

## Collapsed-control test

A matched numerical control was built by placing all six notches at the mean
mapped target while preserving their count and `lambda_soft`.

At `32x32`, `lambda_soft=6`:

- mask correlation with the true six-target implementation: `>0.9999999`
- mean absolute mask difference: `~8e-6`

The same near-identity persists at larger tested grids.

Therefore the current simulation output cannot be attributed to the detailed
spacing of the six selected zeta ordinates. The active implementation is best
described as a broad spectral redistribution/smoothing regularizer whose center
is zeta-indexed.

## Resolution correction

A naive one-dimensional argument using `Delta k = 1/N` would overstate the
resolution problem because the code uses the two-dimensional radial frequency

```
k = sqrt(kx^2 + ky^2)
```

which has denser radial shells.

Exact nearest-shell counting gives five distinct nearest radial shells for the
six targets at `32x32`; `64x64` and `96x96` can assign all six to distinct
nearest radial shells. Thus grid resolution alone is **not** the primary
failure. The dominant failure is the excessive notch width and resulting mask
collapse.

## Consequence for current ablation results

The matched factorial runs remain valid measurements of the implemented
software operator. They do **not** establish a zeta-zero-specific effect.

Until a narrow-band implementation passes matched placebo controls, report the
current layer as:

`ZETA-INDEXED BROAD SPECTRAL REGULARIZER — ZERO SPECIFICITY UNRESOLVED`

## Next falsification gate

A replacement candidate must be tested against controls with the same:

1. number of notches;
2. notch widths and depths;
3. stochastic-noise amplitude;
4. total material-neutral projection;
5. seed, chemistry, geometry and runtime.

At minimum compare:

- narrow zeta targets;
- collapsed mean-target control;
- shifted/equispaced placebo targets;
- noise-only;
- no spectral operator.

Only an effect that separates the zeta-target arm from matched spectral
placebos may be called zero-specific inside the software model. Physical binding
would still remain an independent open question.
