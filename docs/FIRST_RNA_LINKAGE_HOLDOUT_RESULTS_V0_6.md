# First-RNA Linkage / Regioselectivity Holdout v0.6 — Result

Status: `FAIL_MISSING_LINKAGE_STATE`

Date: 2026-09-22  
Execution head: `10d47e19d9c4bb35eb23b32ddc6efb057af8831a`

## Frozen held-out observable

Huang & Ferris (2006), DOI `10.1021/ja061782k`, report under the
1-methyladenine activated montmorillonite condition:

```
A: 74% 3',5' phosphodiester bonds
U: 61% 3',5' phosphodiester bonds
```

These fractions were not used to set the v0.3/v0.5 chain-length calibration
parameters.

## Capability result

The frozen v0.5/v0.3 representation contains no explicit variable for bond
regioisomer identity.

Audited parent fields:

```
ReactiveState:
  unreactive_free
  reactive_free
  chains
  max_relative_conservation_error
```

The frozen profile, state, public v0.5 output, and v0.5 module surface all
failed to expose a 2',5' / 3',5' linkage channel.

Therefore:

```
explicit_linkage_state = false
model_3p5_fraction(A) = NOT AVAILABLE
model_3p5_fraction(U) = NOT AVAILABLE
numerical comparison = NOT RUN
verdict = FAIL_MISSING_LINKAGE_STATE
```

No 50/50 default, empirical copying, or hidden linkage prior was inserted.

## Causal boundary

```
global_rates_retuned = false
base_extension_multipliers_retuned = false
linkage_parameter_added = false
empirical_fraction_copied_into_prediction = false
geometry_used = false
zeta_used = false
replication_used = false
```

## CI / receipt

Dedicated workflow:

- run: `35790087310`
- conclusion: `success`
- dedicated tests: `4 passed`
- artifact id: `10720949692`
- artifact digest:
  `sha256:2b9e4840257510d4f67b49a7ee46e9f69e3889de4a0c9a16946825e2b7d1681a`

Abiogenesis Canon:

- run: `35790087154`
- conclusion: `success`
- repository-wide tests: `115 passed in 4.04s`

## Interpretation

v0.5 can encode the calibrated base-dependent chain-length behavior but does
not contain enough state to address phosphodiester regioselectivity.

This is a model falsification at the representation level, not a runtime
failure.

The next admissible extension is an explicit bond-state channel separating
2',5' and 3',5' linkages. That extension must be calibrated and validated in a
new version; this v0.6 FAIL is immutable provenance.
