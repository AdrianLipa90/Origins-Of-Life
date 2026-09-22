# Exotic Morphogenesis Validation v0.3

Status: `COMPUTATIONAL_CANDIDATE / FACTORIAL_VALIDATED / PHYSICAL_BINDING_OPEN`

Branch: `feat/exotic-biology-framework-v01`

Validated head: `82969cc21f852011e828036118cdbaad65aff37b`

## Question

Can substrate-neutral candidate runtimes form a persistent bounded system without
changing kinetic constants or detection thresholds?

The experiment isolates two structural operators:

```text
information geometry:
  DISTRIBUTED_INFORMATION_BASELINE
  PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE

boundary geometry:
  COLOCATED_BASELINE
  EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE
```

Both redistribution operators preserve their same-state instantaneous source
budget to floating-point tolerance. No kinetic parameter or compartment
threshold is tuned.

## Design

A matched-seed 2x2 factorial was run with seeds `11, 23, 47`.

- ammonia candidate: 3000 steps;
- hydrocarbon candidate: 10000 steps.

The different observation horizons were fixed from earlier reachability
experiments. They change observation duration only, not model dynamics.

## Result: ammonia candidate

### Distributed information + colocated boundary

No closed boundary occurs. Information and boundary eventually saturate the
periodic domain.

### Distributed information + exterior-redistributed boundary

A closed boundary appears transiently, but global information saturation destroys
localization.

Across three seeds:

- mean first closed-boundary step: ~1824;
- mean closed-boundary duration: 59 steps;
- final compartment count: 0/0/0;
- final information threshold occupancy: 100%.

### Peak-redistributed information + colocated boundary

Information remains more heterogeneous, but no closed boundary is formed because
the boundary source remains colocated rather than shell-forming.

### Peak-redistributed information + exterior-redistributed boundary

This is the first condition in the current candidate runtime that retains a
closed boundary through the full observation horizon in all three seeds.

- first closed-boundary steps: 1395, 1409, 1461;
- uninterrupted closed-boundary durations to step 3000: 1606, 1592, 1540;
- final compartment counts: 6, 11, 8;
- no global information saturation by step 3000;
- final information threshold occupancy: ~0.69-0.71;
- final boundary threshold occupancy: ~0.34-0.35.

This is a computational morphogenesis result only. It is not evidence for a
specific ammonia-solvent molecular biology.

## Result: hydrocarbon candidate

The same structural sequence occurs on a slower timescale.

### Distributed information + colocated boundary

No closed boundary occurs. By step 10000 the information field globally exceeds
its declared threshold.

### Distributed information + exterior-redistributed boundary

Closed boundaries are long-lived but transient:

- first closure: ~3908-4311;
- continuous closure: 1053-1398 steps;
- later global information saturation;
- final compartment count: 0/0/0.

### Peak-redistributed information + colocated boundary

Global information saturation is prevented over the 10000-step horizon, but a
closed boundary is still not obtained.

### Peak-redistributed information + exterior-redistributed boundary

All three seeds form long-lived closed boundaries:

- first closure: 4174, 4254, 4334;
- continuous closure: 4919, 5009, 5667 steps;
- no global information saturation by step 10000.

At the final horizon:
- seed 47 retains one closed-boundary candidate;
- seeds 11 and 23 have lost closure before the horizon;
- their final state is `INTERIOR_NOT_CLOSED`, not global information saturation.

Therefore the remaining hydrocarbon bottleneck is boundary-closure persistence,
not information reachability or global information saturation.

## Factorial interpretation

The 2x2 experiment supports a model-internal interaction:

```text
localized information source
        +
exterior boundary source
        ->
substantially longer-lived closed-boundary states
```

Neither operator alone produces the same outcome.

This statement concerns the computational architecture only. Molecular binding
for `I`, `B`, the solvent chemistry and the default dimensionless rates
remains OPEN.

## Validation

At the validated head:

- `Abiogenesis Canon`: 219/219 tests PASS;
- Exotic Relational Benchmark: PASS;
- Exotic Threshold Reachability: PASS;
- Exotic Compartment Persistence: PASS;
- Exotic Source Geometry: PASS;
- Exotic Morphogenesis Source Overlap: PASS;
- Boundary Morphogenesis Trial v0.1: PASS;
- Boundary Redistribution Trial v0.2: PASS;
- Localization Lifecycle v0.1: PASS;
- Hydrocarbon Long Horizon v0.1: PASS;
- Morphogenesis Factorial v0.3: PASS.

## Current gates

### Ammonia computational morphogenesis gate

Current matched-seed/horizon result: `PASS_CANDIDATE`.

Meaning only: all three tested seeds retain at least one closed-boundary
candidate through step 3000 under the combined structural intervention.

It does not promote physical or molecular binding.

### Hydrocarbon computational morphogenesis gate

Current result: `OPEN_PARTIAL`.

All three tested seeds form long-lived closed-boundary candidates, but only one
of three retains closure at step 10000.

The next diagnostic must target shell-loss geometry and must not tune rates or
thresholds.

## Non-claims

This checkpoint does not establish:

- ammonia-based life;
- methane/ethane-based life;
- a molecular information carrier;
- a molecular compartment material;
- experimentally calibrated kinetics;
- correspondence between model steps and physical time;
- abiogenesis in any planetary environment.

It establishes a reproducible computational distinction between global
colocation, transient closure and persistent bounded-state formation.
