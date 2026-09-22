# Hydrocarbon Stability Factorial v0.5

Status: `COMPUTATIONAL_CANDIDATE / NEGATIVE_CAUSAL_RESULT / PHYSICAL_BINDING_OPEN`

Branch: `feat/exotic-biology-framework-v01`

Validated code head: `cc6ac4f20d2a8526365add39501fdbb01ecb48eb`

## Question

Can either of the two failure modes isolated in v0.4 be repaired by a
source-budget-preserving structural intervention without changing kinetic
parameters or detection thresholds?

The factorial isolates:

```text
shell maintenance:
  OFF
  LOW_BOUNDARY_EXTERIOR_MAINTENANCE_CANDIDATE

contractible-island preservation:
  OFF
  STATE_PEAK_ISLAND_PRESERVATION_CANDIDATE
```

The existing morphogenesis pair remains fixed:

```text
PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE
+
EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE
```

Seeds: `11, 23, 47`  
Horizon: `10000` steps  
Grid: `24 x 24`

## Result

The baseline reproduces the v0.4 split:

- seed 11: `CONTRACTIBLE_SHELL_GAP`, loss at 9093;
- seed 23: `CONTRACTIBLE_INTERIOR_LOSS`, loss at 9263;
- seed 47: `PERSISTS_WITHIN_HORIZON`.

### Island-preservation candidate

`STATE_PEAK_ISLAND_PRESERVATION_CANDIDATE` does not solve the persistence
problem.

Compared with baseline:

- seed 11 closure ends earlier: 8368 instead of 9093;
- seed 23 closure ends earlier: 8137 instead of 9263;
- seed 47 still persists, but first closure is later and total closed duration is
  shorter.

The candidate therefore fails its intended causal purpose in this factorial.

### Shell-maintenance candidate

`LOW_BOUNDARY_EXTERIOR_MAINTENANCE_CANDIDATE` also does not solve the
persistence problem.

- seed 11 changes from shell-gap to contractible-interior loss;
- seed 23 remains contractible-interior loss;
- seed 47 persists and ends with more contractible candidates, but this does not
  generalize to the failing seeds.

The candidate therefore also fails its intended causal purpose.

### Combined intervention

Combining shell maintenance with island preservation does not rescue seeds 11 or
23. Both end in `CONTRACTIBLE_INTERIOR_LOSS`.

Seed 47 remains persistent under all tested conditions, indicating that the
dominant unresolved variable is not captured by either intervention.

## Conservation / fairness gates

All 12 factorial cells preserve:

- the modeled material sector;
- the instantaneous information-source budget;
- the instantaneous boundary-source budget.

Maximum source-budget relative errors remain at floating-point scale
(~1e-16). No kinetic coefficient or detection threshold is tuned.

## Causal interpretation

v0.5 falsifies two simple stabilizing hypotheses inside the current model:

```text
more source directed to weak shell sites
    != sufficient closure stabilization

more source directed to existing information peaks
    != sufficient island preservation
```

The next diagnostic must therefore follow the ancestry of contractible
information islands rather than add another control operator.

The unresolved event is:

```text
contractible island
    -> ?
       MERGER_WITH_NONCONTRACTIBLE_BACKGROUND
       THRESHOLD_EROSION
       SPLIT
       DISAPPEARANCE
```

No new stabilizing mechanism should be introduced until this transition is
measured directly.

## Validation

- Abiogenesis Canon: `245/245 PASS`;
- Hydrocarbon Stability Factorial v0.5: PASS;
- artifact digest:
  `sha256:14d3a3c2817951e740572bdc8bbc11c9b5dced0a70bef8c3c4e0cac19cbcf5f3`.

## Non-claims

This checkpoint does not establish hydrocarbon-based life, a physical membrane,
a molecular information carrier, calibrated chemistry, or physical time.

It establishes a reproducible negative causal result inside the computational
candidate architecture.
