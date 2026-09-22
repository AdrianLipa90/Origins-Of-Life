# Zeta Spectral Specificity Preregistration v0.2

Status: `PREREGISTERED / FROZEN BEFORE RUN / APPEND-ONLY`

Date: 2026-09-22  
Parent result: `FAIL_ZERO_SPECIFICITY` from v0.1  
Working branch: `experiments/matched-factorial-ablation-v01`

## Purpose

v0.1 showed that the legacy broad zeta-indexed operator is a strong but
non-specific spectral smoother, while a narrow zeta-target mask did not
separate robustly from a reflected spectral placebo.

v0.2 tests a narrower claim:

> Does the absolute placement of the mapped first-six zeta targets produce a
> reproducible software-model effect that survives several independently
> constructed spectral placebos?

This is a software specificity test only. It does not test physical binding.

## Frozen run design

- scenarios: `A`, `C`
- seeds: integers `201..220` inclusive
- seeds are independent of v0.1 seeds `101..105`
- duration: `120 h`
- grid: `32x32`
- time step: `0.05 h`
- geometry: enabled
- clay: enabled
- RNA initial condition: pre-seeded, matching the current canonical spatial
  simulator boundary
- zeta/noise RNG: isolated deterministic stream already separated from the core
  chemistry/biology RNG
- no parameter tuning is permitted after endpoint results are inspected

Each seed must have an identical pre-operator initial-state digest across all
arms. Any mismatch fails closed.

## Frozen control arms

1. `none`
   - no spectral operator
   - no zeta-side stochastic term

2. `noise_only`
   - identity spectral mask
   - same zero-mean stochastic term as all spectral arms

3. `narrow_zeta`
   - mapped first-six zeta ordinates
   - narrow Gaussian notches
   - neighbor cross-talk target: 1%

4. `narrow_reflected`
   - targets reflected about the zeta-target mean
   - preserves target count, mean, span and all pairwise spacing magnitudes

5. `narrow_shifted`
   - every zeta target shifted by exactly `-0.04 cycles/sample`
   - preserves every internal spacing exactly
   - keeps all targets inside the modeled radial FFT band

6. `narrow_equispaced`
   - six targets equally spaced from the original minimum target to original
     maximum target
   - preserves target count, minimum, maximum and total span but not internal
     spacing

7. `histogram_permuted`
   - starts from the exact `narrow_zeta` mask on the tested grid
   - deterministically permutes attenuation values among Hermitian-conjugate
     Fourier-bin classes
   - keeps DC fixed at 1
   - preserves the complete non-DC attenuation-value multiset exactly
   - permutation seed: `20260922`
   - destroys the original radial target placement

The histogram-permuted arm is the strongest attenuation-matched placebo: any
difference from `narrow_zeta` cannot be explained by total attenuation,
mask-value histogram, or stochastic-noise amplitude alone.

## Frozen endpoints

Primary endpoint:

`n_protocells` at 120 h — connected membrane+polymer components.

Secondary endpoints:

- `protocell_area_pixels`
- `polymer_threshold_pixels`
- `mean_R`
- `max_R`
- `n_polymers`
- `nucleotide_material_total`

Mask diagnostics are model diagnostics, not biological endpoints.

## Frozen primary contrasts

For each scenario and each seed:

```
D_reflected = narrow_zeta - narrow_reflected
D_shifted   = narrow_zeta - narrow_shifted
D_equal     = narrow_zeta - narrow_equispaced
D_hist      = narrow_zeta - histogram_permuted
```

The primary analysis uses the 20 paired differences for `n_protocells`.

## Frozen significance procedure

For each of the four placebo contrasts in each of the two scenarios:

1. compute the exact two-sided paired sign-flip/randomization p-value for the
   mean paired difference;
2. apply Holm correction jointly across the eight primary tests;
3. record mean, median, SD, sign count and the full 20-value paired-difference
   vector.

No asymptotic p-value may replace the exact sign-flip test for the primary gate.

## Frozen verdict rule

`PASS_ZERO_SPECIFICITY` requires all of the following:

1. all four zeta-vs-placebo primary contrasts are Holm-adjusted `p <= 0.05`
   in Scenario A;
2. all four are Holm-adjusted `p <= 0.05` in Scenario C;
3. the mean paired effect has the same sign in both scenarios for every
   placebo family;
4. for every placebo family and both scenarios, the median absolute paired
   difference in connected-protocell count is at least `1` component.

`PARTIAL_ZERO_SPECIFICITY` requires at least two independent placebo families
to satisfy conditions 1--4 in both scenarios.

Otherwise:

`FAIL_ZERO_SPECIFICITY`

Secondary endpoints may explain a result but cannot rescue a failed primary
gate.

## Non-negotiable interpretation boundary

A software specificity PASS would mean only that the chosen zeta-indexed target
placement outperformed the preregistered spectral placebos inside this
simulation.

It would not establish:

- a physical zeta constraint on prebiotic chemistry;
- a biological mechanism;
- empirical abiogenesis;
- any implication for the Riemann hypothesis.

A FAIL is retained as a valid negative result and must not be tuned away.
