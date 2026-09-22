# Ammonia Candidate Runtime v0.1

Status: `COMPUTATIONAL_CANDIDATE / PHYSICAL_BINDING_OPEN`

Runtime code: `AMMONIA_CANDIDATE_V0_1`

This runtime is a dedicated non-RNA, non-lipid computational implementation for
the `AMMONIA_CANDIDATE` profile. It is deliberately separate from the
terracentric `UniversalOriginSimulator`.

## State

The runtime tracks five functional fields:

- `P` — free material / candidate building-block pool;
- `I` — persistent information-carrier proxy mass;
- `B` — compartment-boundary material proxy;
- `E` — environmentally driven energy-availability field;
- `Q` — heritable information-state trait in `[0,1]`.

No claim is made that P, I or B correspond to a specific known ammonia-solvent
molecule.

## Conserved quantity

The material sector is explicitly closed:

```text
M = sum(P + I + B)
```

Assembly transfers material from P to I or B. Degradation returns it to P.
Periodic transport is conservative. Every step checks M and fails loudly if the
residual exceeds numerical tolerance.

Energy `E` is intentionally open because it represents external environmental
throughput.

## Relational life invariants

The runtime operationalizes the framework invariants as computational
observables:

- bounded system -> connected regions where B and I exceed declared thresholds;
- energy throughput -> external E input/loss integral;
- persistent information -> I carrying a local Q state;
- heritable transformation -> neighbor-weighted Q inheritance with mutation;
- selection/differential persistence -> Q-dependent information assembly and
  decay.

Each observable has `physical_binding = OPEN`.

## Parameter status

All default kinetic and transport coefficients are:

`UNVALIDATED_DIMENSIONLESS_CANDIDATE`

They are test parameters for a falsifiable model architecture, not measured
ammonia-biochemistry constants.

## Fail-closed boundaries

- A scenario must explicitly bind `AMMONIA_CANDIDATE`.
- The world solvent must be `NH3`.
- Non-finite or negative state fails loudly.
- Positivity-unstable diffusion fails loudly.
- Material non-conservation fails loudly.
- `HYDROCARBON_CANDIDATE` is not silently routed here.
- The existing universal RNA-like simulator remains
  `TERRACENTRIC_CONTROL_ONLY` for scenario C.

## Non-claims

This runtime does not establish:

- that ammonia-based life exists;
- a molecular identity for the information carrier;
- a molecular identity for the boundary;
- experimentally validated transport constants;
- experimentally validated inheritance chemistry;
- abiogenesis in ammonia.

It is a substrate-agnostic computational candidate that can now be falsified,
ablated and compared without importing RNA or lipid assumptions.
