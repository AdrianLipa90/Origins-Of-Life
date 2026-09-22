# Matched Geometry–Zeta Factorial Ablation v0.1

Status: `CANDIDATE / APPEND-ONLY / NO DNA INTEGRATION`

Base main: `c178d093a529208f97c40b9d01edbd8b173410d7`  
Working branch: `experiments/matched-factorial-ablation-v01`

## Question

Do the experimental geometry and zeta-indexed operators measurably change the
current Origins-Of-Life software model after chemistry, initial state, seed,
grid, time step and run duration are held fixed?

This experiment does **not** ask whether either operator is physically realized
in abiogenesis. Physical binding remains open.

## Design

A 2x2 factorial design is used for every seed:

| Arm | Geometry | Zeta |
|---|---:|---:|
| chemistry_only | 0 | 0 |
| zeta_only | 0 | 1 |
| geometry_only | 1 | 0 |
| full | 1 | 1 |

For geometry-off arms, `topo_strength=0`. All other chemical and environmental
parameters are copied from the same scenario.

For zeta-off arms, `use_zeta_constraints=False`. For zeta-on arms it is forced
`True`, including scenarios whose production default disables it, because this
is an intervention experiment rather than a scenario-default replay.

## Matched-control hardening

Before this branch, the zeta operator consumed the simulator's core RNG stream.
That same stream also drives RNA replication, mutation-like fitness noise,
fragmentation and degradation. Therefore switching zeta on changed later random
draws even apart from the zeta field intervention.

This branch gives zeta its own deterministic RNG stream derived from the scenario
seed. The existing core RNG remains the chemistry/biology stream.

Every arm is initialized independently from the same seed and its pre-operator
chemical fields plus seeded RNA population are hashed. The experiment fails
closed if the four initial-state digests differ.

## Contrasts

For each metric, with C=chemistry_only, Z=zeta_only, G=geometry_only, F=full:

```
zeta_at_zero_geometry     = Z - C
geometry_at_zero_zeta     = G - C
interaction               = F - G - Z + C
full_minus_chemistry      = F - C
```

The interaction term is the non-additive software-model contribution. It is not
a claim of a physical coupling.

## Primary observables

- mean polymer field `mean_R`
- mean membrane field `mean_M`
- RNA population size
- connected protocell-component count
- protocell-positive area
- nucleotide-equivalent material total
- accumulated candidate geometric phase
- candidate Bloch coherence

## Reproduction

```text
python scripts/run_matched_ablation.py --scenario A --seeds 101,102,103,104,105 --hours 12
```

Use `--polymer-free` for the spatial simulator's polymer-free initial condition.
The separate `origins.biology.first_rna` pathway is intentionally not spliced
into this experiment.

## Decision rule

A nonzero numerical contrast is evidence only that the corresponding software
operator changes this model under the tested conditions.

Claims about abiogenesis require, separately:

1. robustness across seeds and numerical resolutions;
2. comparison with matched non-zeta/non-geometric controls;
3. parameter sensitivity and no-go regions;
4. a defensible physical mapping to measurable chemistry;
5. external experimental or observational validation.
