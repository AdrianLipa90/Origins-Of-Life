# First-RNA Local Linkage Context v0.9

Status: `PREREGISTERED REPRESENTATION CALIBRATION / APPEND-ONLY`

Date: 2026-09-22  
Parent result: `v0.8 FAIL_UNIVERSAL_STAGE_SHIFT`

## Purpose

v0.8 falsified an exact pair-independent additive dimer-to-trimer shift.
v0.9 therefore adds the minimum local chemical context required by published
sequence/regioselectivity observations, without fitting another numerical
correction.

## Source constraint

Miyakawa & Ferris (2003), JACS 125, 8202-8208,
DOI `10.1021/ja034328e`, report the categorical ordering

```
Pu(3')Py > Pu(3')Pu = Pu(2')Py > Pu(2')Pu
```

and report that trimer reactivity depends on both:

- the nucleotide attached to the 3' end of the RNA,
- the regiochemistry of the existing phosphodiester bond.

These observations are used as representation/calibration constraints only.
They are not converted into arbitrary kinetic constants.

## v0.9 state

Dimer/local bond context:

```
left_class   in {Pu, Py}
linkage      in {3p5, 2p5}
right_class  in {Pu, Py}
```

Growth context:

```
terminal_class    in {Pu, Py}
previous_linkage  in {3p5, 2p5}
incoming_class    in {Pu, Py}
```

There is deliberately no chain-length field in this minimal local-context
state.

## Frozen ordinal relation

The v0.9 calibration records only the source ordering:

```
{Pu-3p5-Py}
>
{Pu-3p5-Pu, Pu-2p5-Py}
>
{Pu-2p5-Pu}
```

No numeric score, probability, rate, energy, or free pair-specific correction
is assigned to these levels.

## Gate

PASS requires:

1. the four source contexts are represented distinctly,
2. the equality class `Pu(3')Pu = Pu(2')Py` is preserved,
3. the two strict inequalities are preserved,
4. growth context explicitly contains terminal identity, previous linkage, and
   incoming identity,
5. no chain-length term is introduced,
6. no numerical local-context parameter is introduced,
7. all earlier causal boundaries remain false.

PASS verdict:

`PASS_LOCAL_CONTEXT_REPRESENTATION_CALIBRATION`

otherwise:

`FAIL_LOCAL_CONTEXT_REPRESENTATION_CALIBRATION`

## Epistemic boundary

This is a representation repair/calibration, not predictive validation.

A previously inspected 2006 result reports chain-length-invariant structural
proportions within oligo(A) and oligo(U). Because that observation was known
before this preregistration, it is not treated as a held-out v0.9 prediction.

The next scientific gate must use a separate observable/condition that is not
used to define this symbolic context relation.
