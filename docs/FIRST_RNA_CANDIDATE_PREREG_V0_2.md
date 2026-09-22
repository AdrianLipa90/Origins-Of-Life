# First-RNA Candidate Causal Gate v0.2

Status: `PREREGISTERED / CANDIDATE-LENGTH ONLY / APPEND-ONLY`

Date: 2026-09-22  
Parent state: zeta specificity v0.2 = `FAIL_ZERO_SPECIFICITY`

## Causal ordering

The next Origins-of-Life gate is deliberately narrower than "first replicator".

The order is:

```
monomer chemistry
  -> oligomer length reachability
  -> sequence/activity representation
  -> replication chemistry
  -> population dynamics
```

A later stage may not be used to rescue a failure in an earlier stage.

Therefore this experiment disables:

- functional replication;
- logistic replicator growth;
- zeta modulation;
- candidate geometry/Berry modulation.

The only question is whether the current conserved legacy monomer/oligomer
chemistry can produce a 45-nt length candidate from a polymer-free start.

## Why 45 nt is a reference, not a function claim

The reference length is fixed at `45 nt`.

This does **not** mean that an arbitrary 45-mer is a replicator. It is only a
length reference motivated by the experimentally reported QT45 polymerase
ribozyme, a specific 45-nt RNA sequence that can catalyze RNA-templated RNA
synthesis under specific eutectic-ice/triplet-substrate conditions.

Functional replication remains:

`UNRESOLVED_SEQUENCE_AND_ACTIVITY`

## Experimental chemistry boundary

Two external empirical anchors are kept separate from this simulation:

1. Wet-dry cycling of 2',3'-cyclic nucleotides without external activator has
   produced mixed RNA oligomers including detectable 10-mers, with the reported
   10-mer mass fraction around 0.1% under one quaternary-mixture condition.

2. Activated nucleotides on montmorillonite have produced much longer RNA
   oligomers; A/U systems with a phosphate-activating group reached roughly
   40-50 mers in reported experiments.

These anchors show that "unactivated/cyclic wet-dry chemistry" and
"activated-clay chemistry" are not interchangeable mechanisms.

The current legacy simulator does not explicitly track nucleotide activation
state. v0.2 therefore does **not** tune its rate constants to the 40-50-mer
result. It measures the legacy channel as-is and records an activation gap if
the 45-nt candidate is unreachable.

References:

- Huang W, Ferris JP. J Am Chem Soc. 2006;128:8914-8919.
  DOI: 10.1021/ja061782k
- Kawamura K, Ferris JP. J Am Chem Soc. 1994;116:7564-7572.
  DOI: 10.1021/ja00096a013
- High-Yield Prebiotic Polymerization of 2',3'-Cyclic Nucleotides under
  Wet-Dry Cycling. ACS Cent Sci. DOI: 10.1021/acscentsci.5c00488
- QT45 small polymerase ribozyme: PubMed PMID 41678588.

## Frozen simulation

- seeds: `301..320` inclusive
- hours: `500`
- dt: `0.5 h`
- initial nucleotide units: `800`
- temperature: `65 C`
- pH: `7.5`
- legacy clay-catalysis scalar: `7.5`
- concentration boost: `1000`
- wet-dry period: `12 h`
- dry fraction: `0.30`
- geometry contribution: exactly zero
- Berry contribution: exactly zero
- functional replication: disabled

The legacy explicit external monomer input inside `step_oligomer_pool` is
retained unchanged and is reported, not hidden.

## Frozen observables

Primary:

- maximum chain length whose expected count is at least one molecule-equivalent
  at 500 h.

Secondary:

- expected count at or above 10 nt;
- expected count at or above 35 nt;
- expected count at or above 45 nt;
- expected count at or above 50 nt;
- mean oligomer length;
- total nucleotide-equivalent units;
- functional replicator count, which must remain exactly zero.

## Gate

`PASS_LENGTH_REACHABILITY` requires:

1. median max-length-with-count>=1 is at least 45 nt;
2. at least 10 of 20 seeds have expected count >=1 at 45 nt;
3. all runs remain finite and non-negative;
4. functional replicator count remains exactly zero;
5. nucleotide accounting passes the existing conservation checks.

Otherwise:

`FAIL_LENGTH_REACHABILITY`

A FAIL does not mean 45-mers are chemically impossible. It means the present
legacy chemistry channel is insufficient and the missing activation chemistry
must be modeled explicitly rather than hidden in a topology or rate multiplier.
