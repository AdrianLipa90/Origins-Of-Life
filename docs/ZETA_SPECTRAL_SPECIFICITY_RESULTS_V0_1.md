# Zeta Spectral Specificity Results v0.1

Status: `FAIL_ZERO_SPECIFICITY / APPEND-ONLY / NO PHYSICAL NO-GO CLAIM`

Date: 2026-09-22  
Evaluated head: `ced23239a18392fc14315ecb50dc71dc268c973a`

## Question

Does a narrow spectral operator placed at the mapped first six Riemann-zeta
ordinates produce a reproducible model effect that separates from a matched
reflected spectral placebo?

This is a software-model falsification test. A failure here does not establish
that zeta zeros have no physical role in abiogenesis.

## Controls

Five matched modes were run from identical per-seed initial-state digests:

- `none`
- `noise_only`
- `narrow_zeta`
- `narrow_reflected_placebo`
- `legacy_broad`

The narrow placebo reflects the six target frequencies around their mean. It
therefore preserves target count, mean, span and all pairwise spacing
magnitudes while changing absolute spectral placement.

Runs used 5 seeds (101--105), a 32x32 grid, dt=0.05 h and 120 h duration for
Scenarios A and C.

## Primary comparison: narrow_zeta - narrow_reflected_placebo

### Scenario A

| Metric | Mean paired delta | SD | 95% t interval |
|---|---:|---:|---:|
| mean_R | +6.46e-6 | 4.96e-6 | [+3.09e-7, +1.26e-5] |
| max_R | -3.98e-5 | 2.68e-3 | [-3.36e-3, +3.28e-3] |
| polymer-threshold pixels | +1.6 | 3.29 | [-2.48, +5.68] |
| protocell components | -0.8 | 2.68 | [-4.13, +2.53] |
| protocell area pixels | +0.6 | 1.82 | [-1.66, +2.86] |
| nucleotide-material total | +6.62e-3 | 5.08e-3 | [+3.16e-4, +1.29e-2] |

The tiny positive `mean_R` shift is consistent in these five seeds, but it is
about 0.046% of the endpoint mean and does not propagate into a stable
protocell or threshold phenotype.

### Scenario C

| Metric | Mean paired delta | SD | 95% t interval |
|---|---:|---:|---:|
| mean_R | -1.40e-5 | 1.90e-4 | [-2.50e-4, +2.22e-4] |
| max_R | +1.36e-2 | 3.94e-2 | [-3.53e-2, +6.25e-2] |
| polymer-threshold pixels | -1.8 | 12.38 | [-17.17, +13.57] |
| protocell components | -0.4 | 0.55 | [-1.08, +0.28] |
| protocell area pixels | -0.4 | 0.55 | [-1.08, +0.28] |
| nucleotide-material total | +4.52e-5 | 2.61e-4 | [-2.79e-4, +3.69e-4] |

No stable separation from the reflected placebo is present.

## Legacy broad operator

The legacy operator remains numerically potent, but the preceding specificity
audit showed that its six-target mask is almost identical to a collapsed
single-band control and suppresses most non-DC spatial frequencies.

At 120 h in Scenario A the legacy broad operator removes the geometry-only
protocell phenotype (geometry-only: mean 21.8 components; full legacy-broad:
0 across all five seeds). That is a real result of the implemented broad
regularizer, not evidence that the detailed Riemann-zero spacing caused it.

## Control limitation

On the 32x32 radial FFT lattice, the two narrow target sets do not have exactly
the same realized attenuation histogram:

- narrow_zeta mean non-DC mask transmission: ~0.984660
- reflected placebo: ~0.984951
- fraction below 0.9: ~0.02737 vs ~0.03910

Therefore very small endpoint differences cannot be uniquely attributed to
target identity. A future positive specificity claim would require an
attenuation-matched placebo in addition to the reflected-spacing control.

## Verdict

`FAIL_ZERO_SPECIFICITY`

The current evidence supports these statements:

1. the legacy broad zeta-indexed operator strongly changes model spatial
   structure;
2. that legacy effect is dominated by broad spectral smoothing/redistribution;
3. a narrow zeta-target operator does not reproducibly separate from a matched
   reflected placebo on the tested biological observables across A and C;
4. zero-specificity is therefore not established.

This verdict is about the present software implementation and tested parameter
regime only. Physical binding remains open.
