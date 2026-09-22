# First-RNA Pair-Context Holdout v0.8 — Result

Status: `FAIL_UNIVERSAL_STAGE_SHIFT`

Date: 2026-09-22  
Execution head: `46999212ad9c12c1b39221fa111099779fd97970`

## Cross-condition transfer

Calibration condition A/U:

```
dimer 3',5' fraction  = 0.51
trimer 3',5' fraction = 0.61
delta_stage            = +0.10
```

Held-out condition A/C:

```
dimer 3',5' fraction  = 0.49
predicted trimer       = 0.59
observed trimer        = 0.56
prediction error       = +0.03
```

The A/C trimer observation was not used to set the transferred prediction.

## Rounding-only compatibility

Conservative integer-rounding envelopes:

```
A/U stage-shift envelope:
[0.090, 0.110]

A/C predicted trimer envelope:
[0.575, 0.605]

A/C observed trimer envelope:
[0.555, 0.565]
```

The predicted and observed A/C envelopes are disjoint.

Therefore the exact universal-additive dimer-to-trimer shift is rejected at the
point-estimate / rounding-compatibility level.

This is not a statistical-significance claim.

## Interpretation

The dimer-to-trimer increase is present in both binary systems, but the same
additive shift does not transfer exactly from A/U to A/C.

This falsifies a universal pair-independent stage shift. It does not identify a
unique mechanism. The next state extension must allow local monomer-pair /
sequence context, previous-linkage context, or another explicitly represented
condition to modulate linkage selectivity.

Independent qualitative evidence is consistent with this direction: published
montmorillonite studies report that trimer reactivity depends on the nucleotide
at the 3' end and on the regiochemistry of the existing phosphodiester bond.

## Causal boundary

```
pair_specific_parameter_added = false
global_rates_retuned = false
base_extension_multipliers_retuned = false
geometry_used = false
zeta_used = false
replication_used = false
statistical_significance_claim = false
```

## CI and error provenance

Initial dedicated run `35791272666` was technically RED because one unit test
used exact IEEE-754 tuple equality for 0.09. The runner itself completed and
returned `FAIL_UNIVERSAL_STAGE_SHIFT`.

Only that test assertion was changed to `math.isclose`; model code, data,
prediction, gate, and preregistration were unchanged.

Corrected execution:

- head: `46999212ad9c12c1b39221fa111099779fd97970`
- dedicated run: `35791380336`
- dedicated conclusion: `success`
- dedicated tests: `4 passed in 0.41s`
- artifact id: `10722795275`
- artifact digest:
  `sha256:6d06c8e95bfc6a8ea9e800a653e6527a4c10f56681a3770f3defc1b3e473a0c8`
- Canon run: `35791380342`
- Canon conclusion: `success`
- repository-wide tests: `123 passed in 3.11s`

## Next gate

Add explicit local context without introducing a free numerical correction
chosen to reproduce 0.56.

The next test should separate at least:

- terminal purine/pyrimidine identity,
- incoming purine/pyrimidine identity,
- previous 2',5' versus 3',5' linkage.

The 2003 Miyakawa-Ferris ordering provides a categorical constraint for that
state and should be treated as structure evidence, not as a fitted kinetic
constant.
