# Exotic Morphogenesis Validation v0.4

Status: `COMPUTATIONAL_CANDIDATE / CONTRACTIBILITY_VALIDATED / PHYSICAL_BINDING_OPEN`

Branch: `feat/exotic-biology-framework-v01`

Validated code head: `5d83ed6367b7b54ab0593b051337d39caeb82e87`

This checkpoint is append-only relative to v0.3. It does not rewrite the
previous factorial result; it strengthens the definition of a bounded system and
classifies the remaining hydrocarbon closure failures.

## Contractibility gate

A bounded system on the periodic simulation domain must have a contractible
information-rich interior.

A non-contractible component that winds around either periodic axis is therefore
not accepted as a bounded compartment even if a boundary field happens to cover
its local exterior neighbors.

The diagnostic uses a lift of each periodic component to Z^2. Reaching the same
periodic site with a different lifted coordinate identifies non-zero winding.

Synthetic gates confirm:

- a finite interior patch is contractible;
- a stripe winding around axis 0 is non-contractible;
- a stripe winding around axis 1 is non-contractible;
- a non-contractible interior with a complete local shell is rejected.

## Effect on previous morphogenesis result

The v0.3 closure events survive the stronger gate.

The combined structural condition remains:

```text
PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE
+
EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE
```

with frozen kinetic parameters and frozen detection thresholds.

The key result is therefore not an artifact of accepting torus-spanning
information domains as cells.

## Hydrocarbon dropout topology

The hydrocarbon candidate develops a large non-contractible information
component before closure is lost. This by itself is not the dropout mechanism:

- seeds 11 and 23 lose closure later;
- seed 47 still has a non-contractible giant component at step 10000 while a
  separate contractible closed-boundary candidate persists.

The relevant state is coexistence:

```text
non-contractible information background
+
at least one contractible information island
+
complete boundary shell around that island
```

### Seed 11 — CONTRACTIBLE_SHELL_GAP

Last closed step: `9092`  
First loss step: `9093`

At step 9092:
- contractible information components: 2;
- non-contractible information components: 1;
- best accepted interior area: 1 pixel;
- shell: 4 pixels;
- missing shell pixels: 0;
- shell coverage: 1.0.

At step 9093:
- contractible information components: 1;
- non-contractible information components: 1;
- remaining best contractible interior area: 1 pixel;
- shell: 4 pixels;
- missing shell pixels: 1;
- shell coverage: 0.75.

Classification:

`CONTRACTIBLE_SHELL_GAP`

The first loss is therefore a one-site shell defect, not loss of all
contractible information structure.

### Seed 23 — CONTRACTIBLE_INTERIOR_LOSS

Last closed step: `9262`  
First loss step: `9263`

At step 9262:
- contractible information components: 1;
- non-contractible information components: 1;
- best interior: 1 pixel;
- shell: 4/4.

At step 9263:
- contractible information components: 0;
- non-contractible information components: 1;
- no contractible shell candidate remains.

Classification:

`CONTRACTIBLE_INTERIOR_LOSS`

### Seed 47 — PERSISTS_WITHIN_HORIZON

At step 10000:
- contractible information components: 1;
- non-contractible information components: 1;
- accepted contractible interior: 1 pixel;
- shell: 4/4;
- closed-boundary candidate count: 1.

Classification:

`PERSISTS_WITHIN_HORIZON`

## Interpretation

The hydrocarbon persistence bottleneck is now split into two model-internal
failure modes:

```text
A. contractible island survives
   -> shell completeness can fail locally

B. contractible island disappears/merges
   -> only non-contractible information background remains
```

A single generic "homeostasis" operator would confound these mechanisms.
Future interventions must therefore be independently testable as:

1. shell-maintenance candidates;
2. contractible-island-preservation candidates.

Neither intervention may change the existing kinetic constants or detection
thresholds in the corresponding causal test.

## Validation

For the validated code head:

- Abiogenesis Canon: `228/228 PASS`;
- Hydrocarbon Closure Dropout: PASS;
- contractibility synthetic tests: PASS;
- toroidal winding diagnostics: PASS;
- Boundary Redistribution Trial v0.2: PASS;
- the contractibility-gated v0.3 factorial: PASS.

Relevant receipts/artifacts:

- Abiogenesis Canon run: `35724182128`;
- Hydrocarbon Closure Dropout run: `35724182059`;
- dropout artifact: `10693440459`;
- dropout artifact digest:
  `sha256:a44f258677a41f33c27adecf99388d4967e6cdf78767f306f450defe3bdda660`;
- contractibility-gated Factorial run: `35723580140`;
- factorial artifact: `10692640598`;
- factorial digest:
  `sha256:0d38eb92a3b587f6d8dd4cc220e7804532d5ba2a71f33f383d05999758b21c0d`.

## Non-claims

This checkpoint does not establish:

- ammonia-based life;
- hydrocarbon-based life;
- a physical membrane or information molecule;
- calibrated chemical rates;
- a correspondence between simulation steps and physical time;
- that the observed computational topology occurs in Nature.

It establishes a reproducible model distinction between non-contractible
percolation, contractible interior loss, local shell failure and persistent
closed-boundary candidates.
