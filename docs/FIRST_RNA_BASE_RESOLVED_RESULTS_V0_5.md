# First-RNA Base-Resolved Reactive Chemistry v0.5 — Results

Status: `PASS_BASE_RESOLVED_CALIBRATION / CALIBRATION ONLY / APPEND-ONLY`

Date: 2026-09-22  
Execution head: `42206ab2dc512b8490b8735f95dc4ceb5f59d9a7`  
GitHub Actions artifact: `first-rna-base-resolved-v05` (id `10720152805`)  
Artifact SHA256: `4b2d36ebc4819049192e716b8bf5a525c9a411d71157d8a03a3553f55a8eca58`  
Canonical regression gate: `111 passed`

## Parent failure

v0.4 falsified the base-blind v0.3 architecture on the held-out C condition:

```
predicted C tail = 80 nt
empirical C tail = 20-25 nt
```

with conservation still passing.

The repair therefore added explicit base identity rather than changing the
shared conservative reaction skeleton.

## v0.5 calibrated identities

Frozen extension multipliers:

```
A = 1.0
U = 1.0
C = 0.025
G = UNRESOLVED
```

A/U retain the v0.3 activated-clay calibration exactly.

The C multiplier is a coarse calibration parameter chosen after the v0.4
falsification. It is not a measured kinetic constant.

G remains unresolved because the source experiment did not provide a reliable
chain-length determination for G.

## Results

Using the same model-internal tail metric as v0.3/v0.4:

```
A tail = 49 nt at 24 h
U tail = 49 nt at 24 h
C tail = 22 nt at 216 h
G      = UNRESOLVED_EXPERIMENTAL_LENGTH
```

All frozen calibration windows pass:

```
A: 40-55 nt  PASS
U: 40-55 nt  PASS
C: 20-25 nt  PASS
G: unresolved preserved
```

## Integrity

```
global_rates_retuned = false
conservation_pass    = true
geometry_used        = false
zeta_used            = false
replication_used     = false
```

The complete repository test suite remained GREEN:

```
111 passed
```

## Verdict

```
PASS_BASE_RESOLVED_CALIBRATION
```

This is a calibration repair, not a new independent prediction.

The result establishes only that explicit base identity is sufficient for the
coarse-grained conservative model to encode the observed A/U versus C
chain-length separation without altering the global v0.3 rates.

It does not establish that the C multiplier is unique or physically measured.

## Next independent gate

The A/U/C chain-length observations used above are now locked calibration data.

The next test must use an independent observable.

Strong candidates already present in the literature include:

- base-dependent adsorption on montmorillonite;
- phosphodiester linkage/regioselectivity differences;
- feeding-reaction elongation under repeated activated-monomer addition.

The preferred next gate is linkage/regioselectivity because it requires a new
chemical state variable rather than another chain-length fit.
