# First-RNA Explicit Linkage State v0.7

Status: `PREREGISTERED CALIBRATION + PARAMETER-HELD-OUT CHECK / APPEND-ONLY`

Date: 2026-09-22  
Parent result: `v0.6 FAIL_MISSING_LINKAGE_STATE`

## Why this extension exists

v0.6 showed that the frozen v0.5/v0.3 architecture cannot represent
phosphodiester regioisomer identity at all.

v0.7 adds the minimum missing state:

```
B_35    expected bond weight in 3',5' linkages
B_25    expected bond weight in 2',5' linkages
```

No chain-length kinetic rate is changed.

## Source observable

Ertem, Hazen & Dworkin (2007), Astrobiology 7, 715-722,
DOI `10.1089/ast.2007.0138`, reports for the binary A/U montmorillonite
system with phosphorimidazolide-activated monomers:

- prior binary dimer result quoted in the paper: 51% 3',5' linkages,
- trimer result measured in the paper: 61% 3',5' linkages.

The paper also states that each activated monomer was supplied at 0.014 M.

## Frozen calibration

The dimer point estimate is the only linkage calibration datum:

```
p35_dimer = 0.51
```

The zero-order v0.7 hypothesis is length invariance:

```
p35(L) = p35_dimer
```

Therefore the preregistered trimer prediction is:

```
p35_trimer(predicted) = 0.51
```

No trimer value may be used to modify that prediction.

## Held-out point-estimate check

Observed trimer value:

```
p35_trimer(observed) = 0.61
```

The percentages are reported as integer percentages. To avoid inventing a
measurement-error model, v0.7 only tests compatibility under rounding alone:

```
51% -> [0.505, 0.515)
61% -> [0.605, 0.615)
```

If the intervals do not overlap, the length-invariant point-estimate model is
incompatible with the reported rounded values.

Verdict:

`FAIL_LENGTH_INVARIANT_LINKAGE_POINT_ESTIMATE`

if the intervals are disjoint.

This is not a statistical significance claim because the source does not
supply an uncertainty model here.

## Causal boundary

Frozen:

- v0.3 global rates unchanged,
- v0.5 base extension multipliers unchanged,
- no geometry,
- no zeta,
- no replication,
- no sequence/pair interaction term,
- no chain-length dependence in linkage selectivity.

A FAIL identifies the next missing variable class; it must not be repaired
inside v0.7.
