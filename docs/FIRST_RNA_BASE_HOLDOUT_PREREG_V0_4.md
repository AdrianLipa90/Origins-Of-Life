# First-RNA Base-Specific Held-Out Validation v0.4

Status: `PREREGISTERED HELD-OUT VALIDATION / NO RETUNING / APPEND-ONLY`

Date: 2026-09-22
Parent calibration: `PASS_REACTIVE_STATE_CALIBRATION`

## Held-out empirical observation

Huang & Ferris (JACS 2006, DOI 10.1021/ja061782k) reported a strong
base-dependent difference under activated-nucleotide / montmorillonite
conditions:

- A and U activated monomers: RNA oligomers containing about 40-50 monomers in
  one day;
- corresponding C activated monomers: about 20-25-mers after nine days.

The C result was not used in v0.3 calibration.

## Frozen prediction test

The current v0.3 model has no base identity. Therefore v0.4 is intentionally a
hard held-out test of the existing architecture.

No kinetic parameter may be changed.

Reuse exactly the v0.3 `ACTIVATED_CLAY_PROFILE`:

```
initial_units = 10000
initial_reactive_fraction = 1.0
k_activation = 0
k_deactivation = 1e-4 / h
k_nucleation = 1e-10 / h
k_extension = 1e-4 / h
k_hydrolysis = 1e-5 / h
no wet/dry modulation
dt = 0.1 h
max_len = 80
```

Only duration changes from the A/U calibration horizon of 24 h to the
held-out C observation horizon of 216 h (9 days).

No parameter may be made C-specific before this test is evaluated.

## Frozen primary observable

Use the existing v0.3 tail metric:

```
max chain length whose per-length nucleotide mass fraction >= 1e-10
```

This is a model-internal numerical detection criterion. It is not claimed to
equal the experimental detection limit.

## Frozen gate

`PASS_BASE_SPECIFIC_HOLDOUT` requires:

```
20 <= predicted tail length <= 25 nt
```

and nucleotide conservation relative error <=1e-10.

Otherwise:

`FAIL_BASE_SPECIFIC_HOLDOUT`

## Interpretation boundary

A FAIL identifies a missing state variable if the base-blind architecture
cannot represent the held-out C behavior.

After a FAIL, the admissible next step is to add explicit base identity and
base-dependent reaction parameters grounded in external chemistry.

It is not admissible to retune the global A/U-calibrated extension rate and
then call the result a successful prediction.
