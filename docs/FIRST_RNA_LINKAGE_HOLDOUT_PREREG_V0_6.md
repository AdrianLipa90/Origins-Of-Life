# First-RNA Linkage / Regioselectivity Holdout v0.6

Status: `PREREGISTERED STRUCTURAL HOLDOUT / APPEND-ONLY`

Date: 2026-09-22  
Parent: `v0.5 PASS_BASE_RESOLVED_CALIBRATION`

## Question

Can the frozen v0.5 first-RNA architecture predict phosphodiester linkage
regioselectivity without adding a new linkage parameter or retuning any
chain-length calibration parameter?

## Held-out observable

Primary source:

W. Huang and J. P. Ferris, "One-Step, Regioselective Synthesis of up to
50-mers of RNA Oligomers by Montmorillonite Catalysis", JACS 128 (2006),
8914-8919. DOI: 10.1021/ja061782k.

Under the 1-methyladenine activated, montmorillonite-catalyzed conditions
used in that work, the reported fraction of 3',5'-phosphodiester bonds is:

```
A: 0.74
U: 0.61
```

These linkage fractions were not used to set the v0.3/v0.5 chain-length
calibration parameters. This is parameter-held-out validation, not an
analyst-blind experiment.

## Frozen parent model

The parent model is exactly v0.5.

No v0.3 global rate may be changed.
No A/U/C/G extension multiplier may be changed.
No new linkage-selectivity parameter may be introduced in v0.6.

The parent state is the conservative reactive-state representation:

```
U                  free non-reactive nucleotide equivalents
A                  free reactive nucleotide equivalents
C_L                chain population by length L
```

## Capability gate

Before any numerical linkage score is allowed, the frozen model must contain
an explicit state, transition, or output that distinguishes at least:

```
2',5' linkage
3',5' linkage
```

A hidden assumption such as 50/50, random linkage, or copying the empirical
fraction is forbidden.

If the frozen model cannot emit a 3',5' linkage fraction for both A and U
without adding new state or parameters, the preregistered verdict is:

`FAIL_MISSING_LINKAGE_STATE`

and numerical comparison is `NOT_RUN`.

If an explicit linkage state is unexpectedly present, the audit must expose it
and the test is escalated rather than silently reinterpreted.

## Causal boundaries

v0.6 must keep:

- global rates retuned: false
- base extension multipliers retuned: false
- geometry used: false
- zeta used: false
- replication used: false
- empirical linkage fraction copied into model prediction: false

## Interpretation

A structural FAIL is useful evidence. It means v0.5 can reproduce calibrated
chain-length behavior but does not yet contain enough state to address
regioselectivity.

A future linkage-state model must be a separate calibration/extension step and
must not rewrite this result.
