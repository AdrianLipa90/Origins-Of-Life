# First-RNA Functional Gate v0.1

Status: `CANDIDATE / APPEND-ONLY / FUNCTIONAL BINDING OPEN`

Date: 2026-09-22  
Base: `main`

## Problem repaired

The historical `first_rna` runtime promoted an oligomer population into a
"first replicator" when:

- enough oligomers exceeded a length threshold;
- a global mean GC scalar exceeded a threshold;
- a stochastic ignition draw succeeded.

That is a useful phenomenological toy model, but it is not a functional RNA
assay. Length and bulk GC do not establish sequence-specific folding,
polymerase catalysis, template recognition, fidelity, strand separation or
self-copying.

## Current empirical anchor

Gianni et al. reported QT45 in *Science* in 2026
(DOI: `10.1126/science.adt2760`). QT45 is a 45-nt polymerase ribozyme selected
from random-sequence pools. Under mildly alkaline eutectic-ice conditions with
activated trinucleotide substrates it can perform general RNA-templated
synthesis, synthesize its complementary strand and synthesize a copy of itself.
The reported complementary-strand copying fidelity was 94.1% per nucleotide,
while self/complement synthesis yields were about 0.2% over 72 days.

QT45 therefore provides an empirical demonstration that substantial polymerase
function can fit in a small RNA motif. It does **not** imply that a random 45-nt
RNA is a polymerase.

Older montmorillonite experiments demonstrate oligomer production into the
40--50 nt scale under activated-nucleotide laboratory conditions. This makes
the length scale chemically interesting, but still does not bridge length to
function.

## Runtime semantics

The model now separates:

1. `polymerase-sized candidate`: oligomer population reaches at least one
   expected molecule with length >=45 nt;
2. `functional replicator`: requires sequence/folding/activity evidence not
   currently represented by this model.

The default mode is:

`candidate_only`

It can record `first_polymerase_size_candidate_t`, but it cannot set
`first_replicator_t` or create a replicator population.

For historical reproducibility only:

`legacy_phenomenological`

restores the old length+GC stochastic activation. Outputs from this mode are
explicitly labeled `LEGACY_PHENOMENOLOGICAL_MODEL` and must not be presented
as sequence-resolved replication evidence.

## Why no guessed functional probability

No arbitrary "one in N random 45-mers is functional" constant is introduced.
The 2026 QT45 work shows a functional motif and maps its fitness landscape, but
that does not by itself supply a prebiotic prior for spontaneous production
from the chemistry represented here.

A real functional gate should instead consume explicit evidence such as:

- sequence-resolved oligomer populations;
- secondary/tertiary fold constraints;
- substrate-binding and catalytic activity;
- template-complement synthesis;
- fidelity and yield;
- environmental conditions compatible with the assay.

## Remaining gap

The current model can test whether its chemistry reaches polymerase-sized RNA.
It still cannot prove emergence of a functional self-replicating RNA system.

That gap is now explicit rather than hidden behind a stochastic threshold.


## Geometry-control correction

The historical first-RNA kinetic expression also contained a control error:
setting `topo_strength=0` did not remove the geometric contribution. A zero
field maps to `bloch_coherence=1`, while the historical factor was

```
1 + 2 * bloch_coherence
```

so the supposed zero-strength control still multiplied ligation by 3 before
the Berry term.

The hardened runtime therefore uses an explicit causal switch:

- `geometry_mode="off"` (default): Bloch/Berry diagnostics may still be
  computed, but their ligation multiplier is exactly 1;
- `geometry_mode="legacy_candidate"`: restores the historical experimental
  coupling for controlled comparisons.

This prevents a zero-valued synthetic field from being mislabeled as a
chemistry-only control.
