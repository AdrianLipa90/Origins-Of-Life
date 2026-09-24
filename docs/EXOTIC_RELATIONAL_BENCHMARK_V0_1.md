# Exotic Relational Benchmark v0.1

Status: `CANDIDATE / MATCHED-SEED / NO CHEMISTRY RANKING`

Schema: `ORIGINS_EXOTIC_RELATIONAL_BENCHMARK_V0_1`

## Purpose

The dedicated ammonia and hydrocarbon candidate runtimes use different material
graphs and deliberately avoid pretending that their raw kinetic coefficients are
commensurable.

The benchmark therefore compares only observables that exist in both models and
correspond to the shared relational-life contract.

## Shared observables

- absolute material-conservation residual;
- external energy-throughput integral;
- information-proxy mass;
- boundary-proxy mass;
- candidate compartment count;
- candidate compartment area;
- heritable trait mean;
- heritable trait variance.

The benchmark field `ranking_allowed` is always `false`.

## Required matched ablations

Every runtime is tested under the same named interventions:

1. `BASELINE`
2. `NO_SELECTION`
3. `NO_INHERITANCE`
4. `NO_BOUNDARY_ASSEMBLY`
5. `NO_EXTERNAL_ENERGY_INPUT`

Matched random seeds are used across candidate runtimes.

## Hard gates

- Every ablation must preserve the declared material invariant.
- `NO_EXTERNAL_ENERGY_INPUT` must not report positive externally added energy throughput.
- `NO_BOUNDARY_ASSEMBLY` must produce zero boundary mass and no candidate compartments from a zero-boundary initial state.
- `NO_INHERITANCE` must keep Q at zero when the initial Q state is zero.
- Same runtime, seed and ablation must replay deterministically.
- All physical molecular binding remains `OPEN`.

## Interpretation boundary

The benchmark may answer questions such as:

- does the computational architecture conserve its declared material sector?
- is a claimed relational function removed when its operator is ablated?
- is a result reproducible under matched seeds?
- do the two candidate architectures expose the same functional observables?

It may not answer:

- which solvent chemistry is more plausible;
- which candidate biology is more likely to exist;
- whether ammonia or methane/ethane life exists;
- whether raw kinetic constants from the two models are directly comparable.

The purpose is architecture falsification, not a winner table.
