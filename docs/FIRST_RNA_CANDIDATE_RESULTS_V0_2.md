# First-RNA Candidate Causal Gate v0.2 — Results

Status: `FAIL_LENGTH_REACHABILITY / PREREGISTERED NEGATIVE RESULT / APPEND-ONLY`

Date: 2026-09-22  
Preregistration: `docs/FIRST_RNA_CANDIDATE_PREREG_V0_2.md`  
Execution head: `826703bba5f4be5f33d8d73aa79c16e207d5d9d4`  
GitHub Actions artifact: `first-rna-candidate-v02` (id `10703582912`)  
Artifact SHA256: `d915d6f5433993dc0299d5a00637f01a554a1fd5d2e25413f88eb0f0af365e84`  
Canonical regression gate: `98 passed`

## Scope

This gate deliberately tested only the polymer-free monomer/oligomer chemistry
before sequence function or replication.

Disabled by construction:

- zeta spectral operators;
- Bloch/topology contribution;
- Berry contribution;
- functional-replicator ignition;
- logistic replicator population growth.

The 45-nt reference is a length-only benchmark. It is not a claim that an
arbitrary 45-mer is functional.

## Numerical bug exposed before the experiment

The first attempt failed loudly before producing a scientific result.

The legacy oligomer integrator independently capped ligation and hydrolysis
outflow from the same source-bin population. Their combined outflow could
therefore exceed the population available in that bin and produce a negative
state.

The repair at commit `f3294ea038a416fad4df23961e35b475b39c766d`
jointly caps competing source-bin outflows and separately enforces the global
free-monomer budget. A regression test then verifies non-negativity and exact
nucleotide-unit accounting under deliberately extreme competing rates.

This repair changed numerical bookkeeping only. It did not increase a chemical
rate constant or alter the preregistered 45-nt gate.

## Frozen run

- seeds: 301 through 320
- duration: 500 h
- time step: 0.5 h
- temperature: 65 C
- pH: 7.5
- legacy clay-catalysis scalar: 7.5
- concentration boost: 1000
- wet/dry cycle: 12 h
- dry fraction: 0.30
- initial nucleotide-equivalent units: 800
- explicit legacy external monomer input: 5 units/h
- geometry: off
- zeta: off
- replication: off

## Primary result

The preregistered gate required a median maximum chain length of at least
45 nt with expected count >= 1, and at least 10 of 20 seeds with expected
count >= 1 at 45 nt.

Observed:

```
median max length with expected count >= 1: 7 nt
seeds reaching >=1 expected 45-nt candidate: 0 / 20
verdict: FAIL_LENGTH_REACHABILITY
```

Every one of the twenty seeds had the same maximum length of 7 nt under the
expected-count >=1 criterion.

## Endpoint summary

| Observable | Min | Median | Max |
|---|---:|---:|---:|
| mean oligomer length | 2.2324938060 | 2.2324938060 | 2.2324938062 |
| max length with expected count >=1 | 7 | 7 | 7 |
| expected count >=10 nt | 0.0703103190 | 0.0703103190 | 0.0703103191 |
| expected count >=35 nt | 1.52097e-14 | 1.52099e-14 | 1.52115e-14 |
| expected count >=45 nt | 5.99453e-18 | 6.01962e-18 | 6.02942e-18 |
| expected count >=50 nt | 1.13670e-18 | 1.16272e-18 | 1.17440e-18 |

The seed-to-seed spread is negligible at these endpoint statistics. The failure
is therefore not a rare-event miss caused by an unlucky subset of seeds.

## Material accounting

The legacy chemistry includes a declared external input of 5 nucleotide units
per hour.

Over 500 h:

```
initial units       = 800
external input      = 2500
expected final      = 3300
observed final      = 3300 within ~4e-12 absolute numerical error
```

Thus the length failure is not caused by hidden material loss. Even after more
than quadrupling the represented nucleotide inventory, the model remains
concentrated in short chains.

## Functional-replication boundary

For every seed:

```
functional_replication_status = UNRESOLVED_SEQUENCE_AND_ACTIVITY
final_functional_replicators  = 0
functional_replicator_t        = null
```

This is intentional. Candidate length must be established before sequence
function or replication can be evaluated.

## Scientific interpretation

The present legacy monomer/oligomer channel is insufficient to reach the 45-nt
candidate-length benchmark under the frozen conditions.

This does not imply that 45-nt RNA is chemically impossible. It means the
current model is missing a mechanism needed to generate a sufficiently long
tail.

The most important unresolved distinction is nucleotide activation chemistry.
Published activated-monomer/montmorillonite experiments and unactivated or
cyclic-nucleotide wet/dry experiments are different chemical regimes and must
not be collapsed into one scalar "catalysis" multiplier.

Therefore the next admissible model step is:

`EXPLICIT ACTIVATION-STATE CHEMISTRY`

not a larger topology coefficient, not a zeta term, and not retuning
`K_LIG_BASE` until a 45-mer appears.

## Next gate

Introduce explicit pools for at least:

- unactivated/free nucleotide;
- activated nucleotide;
- chain-bound nucleotide;

with conservative transitions and an explicit activation/deactivation or
hydrolysis channel.

The next experiment should test whether an empirically anchored activation
regime can extend the chain-length distribution while preserving nucleotide
accounting. Only after candidate-length reachability passes should sequence and
replication function be added.
