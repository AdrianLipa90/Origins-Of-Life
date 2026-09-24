# Hydrocarbon Candidate Runtime v0.1

Status: `COMPUTATIONAL_CANDIDATE / PHYSICAL_BINDING_OPEN`

Runtime code: `HYDROCARBON_CANDIDATE_V0_1`

This runtime is a dedicated non-RNA, non-lipid computational implementation for
the `HYDROCARBON_CANDIDATE` profile. It is separate from both the water
reference runtime and the ammonia candidate runtime.

## State graph

The material sector is intentionally different from the NH3 candidate:

```text
S <-> A -> I
      \-> B
```

where:

- `S` is dissolved/non-polar precursor material;
- `A` is an explicit interfacial/aggregate-bound precursor reservoir;
- `I` is a persistent information-carrier proxy;
- `B` is a generic persistent boundary/compartment proxy;
- `E` is environmentally driven energy availability;
- `Q` is a heritable information-state trait.

The explicit `A` reservoir is a model-level way to represent interface-mediated
assembly without assuming a terrestrial membrane or a specific azotosome.

## Conserved quantity

```text
M = sum(S + A + I + B)
```

Interface capture/release, information assembly/decay, boundary assembly/decay
and transport are all checked against this material invariant.

Energy `E` remains externally open and is tracked as throughput.

## Relational life invariants

The same substrate-neutral contract is operationalized:

- bounded system -> generic B/I connected regions;
- energy throughput -> external E input/loss;
- persistent information -> I with local Q state;
- heritable transformation -> Q inheritance with mutation;
- selection/differential persistence -> Q-dependent assembly and decay.

Every physical binding remains `OPEN`.

## Parameter status

All default coefficients are:

`UNVALIDATED_DIMENSIONLESS_CANDIDATE`

They are computational test parameters, not measured Titan chemistry constants.

## Explicit non-claims

The runtime does not claim:

- that methane/ethane-based life exists;
- that a specific azotosome is viable or required;
- that B corresponds to a known membrane molecule;
- that I corresponds to a known information polymer;
- that the interface template is a molecular surface model;
- that the default coefficients are empirical.

## Separation from other runtimes

- `UniversalOriginSimulator` remains `TERRACENTRIC_CONTROL_ONLY` for scenario D.
- `AMMONIA_CANDIDATE_V0_1` uses a different material graph and has no explicit A reservoir.
- `HYDROCARBON_CANDIDATE_V0_1` must only be created from a world bound to
  `HYDROCARBON_CANDIDATE`.
