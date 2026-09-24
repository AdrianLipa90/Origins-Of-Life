# First-RNA Reaction-Regime Representation Gate v0.3

Status: `PREREGISTERED / REPRESENTATION BEFORE KINETIC FIT / APPEND-ONLY`

Date: 2026-09-22  
Parent result: `FAIL_LENGTH_REACHABILITY` from First-RNA Candidate v0.2

## Correction to the v0.2 next-step label

v0.2 concluded that the legacy monomer/oligomer channel is missing chemistry
needed to generate a long tail and proposed explicit activation-state chemistry
as the next gate.

That label was too narrow.

Two experimentally demonstrated routes show that long/short RNA-like oligomer
distributions can arise in chemically distinct microenvironments:

1. Mild-alkaline wet-dry cycling of 2',3'-cyclic NMPs at room temperature can
   give high total polymerization yield but a comparatively short detected tail.

2. Hot acidic wet-dry cycling of ordinary nucleotide monophosphates can generate
   much longer oligomers without an externally supplied imidazolide activator.

Therefore the missing model variable is generalized to:

`REACTION_MICROENVIRONMENT / REACTIVE_STATE`

Activation state is one possible component of that state, not the whole state.

This correction is append-only. It does not rewrite the v0.2 result.

## Empirical anchor A — Caimi et al. 2025

Source:
Federico Caimi et al.,
"High-Yield Prebiotic Polymerization of 2',3'-Cyclic Nucleotides under Wet-Dry Cycling",
ACS Central Science 11 (2025) 1546-1557.
DOI: 10.1021/acscentsci.5c00488

Frozen pH-11 AUGC anchor:

- monomer state: 2',3'-cyclic phosphate NMPs;
- mixture: A/U/G/C;
- initial total concentration: 50 mM;
- temperature: 23 C;
- wet-dry cycling: 10 cycles;
- period: approximately 24 h/cycle;
- initial pH: 11;
- total oligomerization yield: 36%;
- detected AUGC 8-mer concentration: 30 uM;
- corresponding reported 8-mer mass fraction: 0.5%;
- maximum reported AUGC length at pH 11: 8 nt.

A separate pH-10 condition reported longer 10-mers and must not be silently
mixed into the pH-11 anchor.

## Empirical anchor B — Song et al. 2024

Source:
Xiaowei Song, Povilas Simonis, David Deamer, Richard N. Zare,
"Wet-dry cycles cause nucleic acid monomers to polymerize into long chains",
PNAS 121 (2024) e2412784121.
DOI: 10.1073/pnas.2412784121

Frozen UMP distribution anchor:

- monomer state: ordinary uridine monophosphate, free-acid/acidic regime;
- initial concentration: approximately 10 mM;
- dry-film temperature: 85 C;
- water-solid/air interface concentration by evaporation;
- distribution comparison: after two wet-dry cycles;
- detected UMP-chain mean length: 16.3 nt;
- detected UMP-chain SD: 10.5 nt;
- detected maximum: 53 nt;
- 35- and 43-mers were explicitly identified by mass spectrometry.

The MS intensity/cluster distribution is an approximate abundance proxy because
ionization efficiency may depend on oligomer length. v0.3 records that limitation.

## Downstream functional anchor — QT45

QT45 is not a spontaneous-polymerization calibration point.

Source:
Gianni et al.,
"A small polymerase ribozyme that can synthesize itself and its complementary strand",
Science 391 (2026) 1022-1028.
DOI: 10.1126/science.adt2760

It is stored only as a downstream function reference:

- 45 nt;
- specific selected sequence;
- polymerase activity with activated trinucleotide-triphosphate substrates;
- mildly alkaline eutectic ice;
- approximately 94.1% per-nucleotide fidelity for complementary-strand synthesis;
- approximately 0.2% self-copy yield over 72 days under the reported conditions.

An arbitrary 45-mer is not QT45.

## Representation gate

v0.3 does not fit kinetic constants.

It must first prove that the software can represent distinct experimental
regimes without collapsing them into one scalar `k_lig`.

Required PASS conditions:

1. Caimi-2025 pH-11 signature binds uniquely to the Caimi anchor.
2. Song-2024 hot-acid signature binds uniquely to the Song anchor.
3. The legacy v0.2 signature
   (`65 C / pH 7.5 / generic clay scalar / unspecified monomer reactive state`)
   binds to no empirical anchor and returns `UNBOUND`.
4. Polymer-length mass fractions are invariant to multiplying all model counts
   by an arbitrary positive scale.
5. Polymerized mass fraction is computed from nucleotide mass, not chain count.
6. QT45 remains in a distinct downstream functional-anchor registry.
7. No kinetic rate is introduced or fitted by this gate.

Verdict:

`PASS_REACTION_REGIME_REPRESENTATION`

only if all seven conditions pass; otherwise fail closed.

## What comes after PASS

Only after this gate may a conservative population-balance kernel be calibrated.

Calibration and validation must remain separated:

- fit a kernel to a subset of summary observables;
- test withheld tail observables without retuning;
- preserve nucleotide-equivalent mass;
- keep the two empirical regimes separate unless a single mechanistic law
  independently predicts both.

No topology, zeta, or replication term may be used to rescue a chemistry fit.
