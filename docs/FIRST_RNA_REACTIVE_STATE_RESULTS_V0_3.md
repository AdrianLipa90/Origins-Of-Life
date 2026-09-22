# First-RNA Reactive-State Calibration v0.3 — Results

Status: `PASS_REACTIVE_STATE_CALIBRATION / CALIBRATION ONLY / APPEND-ONLY`

Date: 2026-09-22  
Execution head: `5d8ff8693ff6e7632def683fb72eb6e68cbee826`  
GitHub Actions artifact: `first-rna-reactive-state-v03` (id `10720412112`)  
Artifact SHA256: `5e059b39f4d69129e85b26249d55c2836687209213516c3388afa0c3d51a48e5`  
Canonical regression gate: `103 passed`

## What changed relative to v0.2

v0.2 failed because the legacy single-`monomer_pool` channel did not generate
a 45-nt candidate tail under the frozen conditions.

v0.3 does not raise a topology coefficient, add zeta, or turn on replication.
Instead it introduces an explicit coarse-grained reactive-state split:

```
U  = free non-reactive nucleotide equivalents
A  = free reactive/activated nucleotide equivalents
C_L = chain population at length L >= 2
```

with conservative transitions:

```
U -> A
A -> U
2 A -> C_2
C_L + A -> C_(L+1)
C_L -> C_(L-1) + U
C_2 -> 2 U
```

No hidden material source is present.

## Calibration anchors

The v0.3 fixtures were frozen against two distinct empirical regimes before the
first v0.3 CI execution.

### Cyclic wet-dry anchor

Reference: Caimi et al., ACS Central Science (2025),
DOI `10.1021/acscentsci.5c00488`.

Broad frozen target:

```
3e-4 <= mass fraction in chains >=10 nt <= 3e-3
```

Observed:

```
mass fraction >=10 nt = 9.390845984810767e-4
                         = 0.09390845984810767%
```

Result: `PASS`.

### Activated-clay long-tail anchor

Reference: Huang & Ferris, JACS (2006),
DOI `10.1021/ja061782k`.

Long-tail abundance context:
DOI `10.1016/j.jasms.2006.05.012`.

Frozen targets:

```
1e-9 <= mass fraction in chains >=40 nt <= 1e-6
40 <= max length with per-length mass fraction >=1e-10 <= 55 nt
```

Observed:

```
mass fraction >=40 nt = 1.4670292110904008e-7
tail length            = 49 nt
```

Result: `PASS`.

## Conservation

Maximum relative nucleotide-unit drift over both calibration trajectories:

```
1.0913936421273939e-15
```

The gate required <= `1e-10`.

Result: `PASS`.

All state variables remained finite and non-negative.

## Causal boundary

By construction:

```
geometry_used    = false
zeta_used        = false
replication_used = false
```

Therefore the calibration cannot be attributed to any of those later layers.

## Verdict

```
PASS_REACTIVE_STATE_CALIBRATION
```

This PASS has a deliberately narrow meaning:

> A nucleotide-reactive-state plus conservative chain-growth architecture can
> represent, after coarse calibration, both a wet-dry short-oligomer regime and
> a sparse activated-clay long-tail regime without invoking geometry, zeta, or
> replication.

It does **not** establish that the fitted normalized rates are measured physical
constants, unique, prebiotically correct, or predictive.

## What is now closed and what remains open

Closed at software-model level:

- explicit separation of non-reactive and reactive free nucleotide pools;
- conservative nucleation / extension / terminal hydrolysis bookkeeping;
- ability to represent the order-of-magnitude difference between the two
  calibration regimes;
- nucleotide-unit conservation.

Still open:

- independent predictive validation;
- base-specific kinetics;
- linkage regioselectivity;
- explicit chemical identity of activation intermediates;
- surface adsorption/desorption;
- sequence representation;
- catalytic activity;
- template copying;
- self-replication.

## Next gate

The two calibration anchors are now locked and may not be reused for fitting.

The next experiment must be held out from v0.3 calibration. Candidate targets:

1. base-specific behavior, e.g. the reported slower/shorter C regime under
   activated-clay conditions;
2. a changed activation group or clay-feeding condition;
3. linkage-regioselectivity fractions.

The preferred next gate is base-specific holdout because it attacks a major
missing degree of freedom directly.
