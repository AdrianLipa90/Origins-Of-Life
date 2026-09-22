# First-RNA Base-Specific Held-Out Validation v0.4 — Results

Status: `FAIL_BASE_SPECIFIC_HOLDOUT / PREREGISTERED NEGATIVE RESULT / APPEND-ONLY`

Date: 2026-09-22  
Execution head: `8fea1612658fdbb9fbbe8975d0ad53016d8bcf5c`  
GitHub Actions artifact: `first-rna-base-holdout-v04` (id `10720412781`)  
Artifact SHA256: `1862060a2e8a07912f1a40e676a884d94c25e79ab605189318a9274417cb755e`  
Canonical regression gate: `106 passed`

## Held-out target

The C-base result from Huang & Ferris (JACS 2006,
DOI `10.1021/ja061782k`) was not used in v0.3 calibration.

Held-out empirical window:

```
20-25 nt after 9 days
```

The v0.4 test reused the v0.3 activated-clay fixture without changing any
kinetic parameter. Only the runtime changed from 24 h to 216 h.

## Result

Observed from the base-blind model:

```
predicted tail length = 80 nt
empirical window       = 20-25 nt
status                 = FAIL_BASE_SPECIFIC_HOLDOUT
```

The predicted tail hit the model's configured maximum chain length.

Additional endpoint values:

```
mass fraction >=10 nt = 1.3685236026598374e-2
mass fraction >=30 nt = 1.3307401446234875e-2
mass fraction >=40 nt = 1.2972283703960643e-2
```

## Numerical integrity

The failure is not numerical instability.

```
max relative conservation error = 2.0008883439005247e-15
conservation_pass                = true
parameters_retuned               = false
```

The complete repository test suite remained GREEN:

```
106 passed
```

## Interpretation

This is a clean model falsification.

The v0.3 architecture can be calibrated to represent:

- the wet-dry short-oligomer anchor;
- the sparse A/U activated-clay long tail.

But it cannot predict the held-out C behavior because the model contains no
base identity and therefore has no mechanism by which C can polymerize
differently from A/U.

The failure must not be repaired by changing the global v0.3
`k_extension`, `k_hydrolysis`, or nucleation rate.

The admissible next causal step is:

`EXPLICIT BASE-RESOLVED REACTIVE CHEMISTRY`

with separate A/U/G/C reactive-state parameters or experimentally motivated
transition laws.

## Consequence

The current coarse-grained state

```
U <-> A -> C_L
```

is sufficient as a mass-conservative reaction skeleton, but not as a
base-independent predictive chemistry.

The minimum next state must distinguish nucleotide identity:

```
U_A, U_U, U_G, U_C
A_A, A_U, A_G, A_C
chain composition / terminal identity
```

or an equivalent representation that can generate base-dependent extension,
deactivation, adsorption, and hydrolysis behavior.

Sequence-level function and replication remain downstream and must not be
introduced until this chemical holdout problem is repaired.
