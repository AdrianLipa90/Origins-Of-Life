# Origins/NOEMA v0.5 — DNA Four-Block Relation Results

## Status

- NOEMA current state: `NOEMA-20260517T024500-v05dnafourblocks`
- Canon allowed: `False`
- Model type: deterministic semantic relation model, not quantum chemistry and not empirical abiogenesis proof.
- Update snap directory: `NOEMA_UPDATE_SNAP/`

## Core result

The model does not treat A, C, G and T as four isolated symbols. It projects the alphabet into two complementary axes plus a purine/pyrimidine symmetry:

- A-T: two-hydrogen-bond proxy, labile information axis; mean semantic affinity `0.933333`; mean stability `0.9`.
- C-G: three-hydrogen-bond proxy, stiff clamp axis; mean semantic affinity `1.0`; mean stability `1.0`.

## Metrics

- ordered_relation_count: `16.0`
- complement_ordered_relation_count: `4.0`
- unordered_complement_axis_count: `2.0`
- complement_selectivity_proxy: `0.791667`
- grammar_compression_proxy: `0.875`
- identity_retention_proxy: `0.911833`
- transition_readiness_proxy: `0.909333`
- alphabet_closure_proxy: `1.0`
- class_symmetry_proxy: `1.0`

## Interpretation

The four DNA bases are not modeled as four independent blocks. They form two complementary axes, A-T and C-G. The A-T axis is the more labile two-bond information splitter; the C-G axis is the stiffer three-bond clamp. Purine/pyrimidine opposition closes the alphabet, and semantic compression preserves the relation grammar as two axes plus class symmetry.

## Boundary

- semantic_relation_model_claimed: `True`
- empirical_abiogenesis_proof_claimed: `False`
- quantum_chemistry_claimed: `False`
- thermodynamic_dna_folding_claimed: `False`
