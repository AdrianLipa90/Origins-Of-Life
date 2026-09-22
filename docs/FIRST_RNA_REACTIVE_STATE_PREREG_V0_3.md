# First-RNA Reactive-State Calibration v0.3

Status: `PREREGISTERED CALIBRATION / NO PREDICTIVE VALIDATION CLAIM / APPEND-ONLY`

Date: 2026-09-22  
Parent gate: `FIRST_RNA_CANDIDATE_V0_2 = FAIL_LENGTH_REACHABILITY`

## Purpose

v0.2 showed that the legacy single-`monomer_pool` chemistry remains concentrated
in short chains even with exact nucleotide accounting. v0.3 introduces an
explicit reactive-state split instead of retuning `K_LIG_BASE`.

The coarse-grained state is:

```
U  = free, non-reactive nucleotide-equivalent pool
A  = free, reactive/activated nucleotide-equivalent pool
C_L = chain count at length L, L >= 2
```

Permitted transitions are:

```
U -> A                 activation / reactive-state entry
A -> U                 deactivation
2 A -> C_2             chain nucleation
C_L + A -> C_(L+1)     chain extension
C_L -> C_(L-1) + U     terminal hydrolytic loss, L > 2
C_2 -> 2 U             dimer hydrolysis
```

Every transition is nucleotide-unit conservative.

No zeta, Bloch/Berry geometry, replication ignition, logistic population growth,
sequence fitness, or hidden material source is permitted in this gate.

## Empirical anchors

This is a calibration test, not an independent validation.

### Anchor A — cyclic nucleotide wet/dry regime

Caimi et al. (ACS Central Science, 2025; DOI 10.1021/acscentsci.5c00488)
reported spontaneous polymerization of 2',3'-cyclic nucleotides under repeated
wet-dry cycling without an external activating reagent. In a mixed-nucleotide
system the reported product distribution included approximately 0.1% 10-mer.

The frozen broad calibration envelope is:

```
mass fraction in chains >=10 nt after 240 h:
3e-4 <= f_10 <= 3e-3
```

The one-order-of-magnitude-wide envelope is intentional: the model is not a
base-resolved reproduction of the experiment.

### Anchor B — activated nucleotide + montmorillonite regime

Huang & Ferris (JACS, 2006; DOI 10.1021/ja061782k) reported that A and U
5'-nucleotides activated with 1-methyladenine formed RNA oligomers containing
40-50 monomers within one day on montmorillonite.

A related MALDI study reported that >=30-mer products can occur at less than
1e-4% of total RNA (fraction <1e-6), so the long-chain tail need not carry a
large mass fraction.

The frozen broad calibration envelope is:

```
24 h activated-clay regime:
1e-9 <= mass fraction in chains >=40 nt <= 1e-6
40 <= max length whose per-length mass fraction >=1e-10 <= 55
```

The lower bounds are computational non-degeneracy thresholds, not measured
experimental yields.

## Frozen coarse-grained profiles

These are calibration fixtures. They are not asserted as measured rate
constants.

All rates are per hour in the model's normalized concentration coordinates.

Shared:

```
initial nucleotide units = 10000
initial reactive fraction = 1.0
k_deactivation = 1e-4
k_hydrolysis = 1e-5
k_nucleation = 1e-10
max chain length = 80
dt = 0.1 h
```

Wet/dry cyclic profile:

```
duration = 240 h
cycle = 24 h
dry fraction = 0.5
k_extension_wet = 5e-7
dry extension multiplier = 10
dry hydrolysis multiplier = 0.1
```

Activated-clay profile:

```
duration = 24 h
no wet/dry modulation
k_extension = 1e-4
```

## Frozen gate

`PASS_REACTIVE_STATE_CALIBRATION` requires all of:

1. wet/dry >=10-nt mass fraction lies in [3e-4, 3e-3];
2. activated-clay >=40-nt mass fraction lies in [1e-9, 1e-6];
3. activated-clay tail length at per-length mass fraction >=1e-10 lies in
   [40, 55] nt;
4. total nucleotide units are conserved to relative error <=1e-10 at every
   recorded endpoint;
5. every state variable remains finite and non-negative;
6. geometry, zeta and replication remain absent by construction.

Otherwise `FAIL_REACTIVE_STATE_CALIBRATION`.

A PASS means only that the explicit reactive-state architecture can represent
both empirical regimes at coarse-grained distribution level after calibration.
It is not evidence that the fitted normalized rates are unique, prebiotically
correct, or predictive.

## Next validation after a PASS

Do not use the two calibration anchors again.

The next independent test should be a held-out chemical condition, preferably a
base-specific or altered-activation experiment. Base identity and linkage
regioselectivity remain OPEN.
