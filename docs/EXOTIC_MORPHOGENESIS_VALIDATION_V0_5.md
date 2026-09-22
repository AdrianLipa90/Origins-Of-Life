# Exotic Morphogenesis Validation v0.5

Status: `COMPUTATIONAL_CANDIDATE / STABILITY_INTERVENTIONS_REJECTED / PHYSICAL_BINDING_OPEN`

Branch: `feat/exotic-biology-framework-v01`

Validated code head: `cc6ac4f20d2a8526365add39501fdbb01ecb48eb`

This checkpoint is append-only relative to v0.4. It tests two independently
motivated hydrocarbon stability interventions after the contractibility gate
split closure loss into shell failure and contractible-island loss.

## Question

Can either of these source-budget-preserving interventions extend closed-boundary
persistence without changing kinetic constants or detection thresholds?

1. `LOW_BOUNDARY_EXTERIOR_MAINTENANCE_CANDIDATE`
2. `STATE_PEAK_ISLAND_PRESERVATION_CANDIDATE`

The fixed background condition is:

```text
PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE
+
EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE
```

The experiment is a matched-seed 2x2 factorial over seeds `11, 23, 47` with
a fixed horizon of 10000 steps.

## Hard constraints

The trial preserves:

- the existing kinetic parameters;
- the existing detection thresholds;
- the instantaneous information-source budget;
- the instantaneous boundary-source budget;
- the total modeled material sector.

No chemistry ranking is permitted and molecular binding remains `OPEN`.

## Result

### Baseline stability modes OFF/OFF

Closed-step counts:

- seed 11: 4919
- seed 23: 5009
- seed 47: 5667

Mean: ~5198.3 closed steps.

Only seed 47 remains closed at step 10000.

### Island preservation only

Closed-step counts:

- seed 11: 3371
- seed 23: 3258
- seed 47: 5111

Mean: ~3913.3 closed steps.

Relative to matched baseline seeds, the intervention changes closed duration by:

- seed 11: -1548 steps
- seed 23: -1751 steps
- seed 47: -556 steps

Mean matched change: -1285 steps.

This candidate therefore does not preserve contractible islands in the tested
dynamics. It is rejected as a stability solution for this checkpoint.

### Shell maintenance only

Closed-step counts:

- seed 11: 4736
- seed 23: 4810
- seed 47: 5610

Mean: ~5052.0 closed steps.

Matched changes versus baseline:

- seed 11: -183 steps
- seed 23: -199 steps
- seed 47: -57 steps

Mean matched change: ~-146.3 steps.

The intervention changes the dropout taxonomy in seeds 11/23 toward
`CONTRACTIBLE_INTERIOR_LOSS`, but does not improve persistence. It is therefore
not promoted as a stability solution.

### Both interventions

Closed-step counts:

- seed 11: 3388
- seed 23: 3728
- seed 47: 5139

Mean: ~4085.0 closed steps.

Matched mean change versus baseline: ~-1113.3 steps.

Again, only seed 47 remains closed at the observation horizon.

## Conservation and implementation checks

Across all factorial rows:

- material residual is at most approximately `2.3e-13`;
- information stability-source budget relative error is below approximately
  `4.8e-16`;
- boundary stability-source budget relative error is below approximately
  `4.9e-16`.

The negative result therefore occurs while the declared source and material
budgets remain numerically closed to floating-point tolerance.

## Interpretation

The v0.4 causal split was useful, but the first candidate corrections are not
solutions.

In this model:

```text
local shell deficit redistribution
    != sufficient shell persistence

state-peak source reinforcement
    != contractible-island preservation
```

The state-peak intervention is actively counterproductive over the tested
matched seeds and horizon.

The remaining bottleneck is therefore not licensed to be described as a generic
"homeostasis" problem. The next diagnostic must identify the event-level cause
of contractible-island disappearance and shell rupture without changing rates,
thresholds or source budgets.

## Validation

At the validated head:

- Abiogenesis Canon: `245/245 PASS`;
- Hydrocarbon Stability Factorial v0.5: PASS;
- all previously active exotic-biology workflows on the head: PASS.

Factorial workflow run: `35725276518`  
Factorial artifact: `10692637781`  
Artifact digest:
`sha256:14d3a3c2817951e740572bdc8bbc11c9b5dced0a70bef8c3c4e0cac19cbcf5f3`

Abiogenesis Canon run: `35725276516`.

## Gate

Hydrocarbon computational stability gate:

`OPEN_FAIL_CANDIDATE_INTERVENTIONS`

Meaning:

- reproducible long-lived closure exists;
- contractibility-gated closure remains valid;
- neither tested stability intervention improves matched-seed persistence;
- no stability intervention is promoted;
- the causal bottleneck remains open.

## Non-claims

This checkpoint does not establish:

- hydrocarbon-based life;
- ammonia-based life;
- a physical membrane or information molecule;
- calibrated chemical rates;
- a correspondence between simulation steps and physical time;
- that these computational morphogenesis operators occur in Nature.
