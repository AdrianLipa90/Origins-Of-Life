"""Four-block DNA relation model for Origins/NOEMA v0.5.

This layer asks what the semantic-compression model says when the nucleic
alphabet is reduced to the four DNA bases A, C, G and T. It is a deterministic
relation model, not a quantum chemistry calculation and not an empirical proof
of abiogenesis.

The model treats the four bases as a small typed graph:

* A and G are purines.
* C and T are pyrimidines.
* A complements T with a lower-stability two-hydrogen-bond proxy.
* C complements G with a higher-stability three-hydrogen-bond proxy.

The output is a NOEMA-facing invariant: not "four isolated letters", but two
conjugate complement axes plus a purine/pyrimidine class symmetry and a
stability gradient.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite
from typing import Any, Dict, List, Mapping, Tuple

SCHEMA = "ORIGINS_NOEMA_DNA_FOUR_BLOCK_RELATION_V0_5"
BASE_ORDER = ("A", "C", "G", "T")
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
BASE_CLASS = {"A": "purine", "G": "purine", "C": "pyrimidine", "T": "pyrimidine"}
RING_COUNT = {"A": 2, "G": 2, "C": 1, "T": 1}
PAIR_H_BONDS = {frozenset(("A", "T")): 2, frozenset(("C", "G")): 3}
PAIR_AXIS = {frozenset(("A", "T")): "AT_labile_axis", frozenset(("C", "G")): "CG_stiff_axis"}


def _clamp01(x: float) -> float:
    if not isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


@dataclass(frozen=True)
class DNABaseBlock:
    symbol: str
    chemical_class: str
    ring_count: int
    complement: str
    semantic_role: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaseRelation:
    source: str
    target: str
    relation_kind: str
    complementarity: float
    purine_pyrimidine_opposition: float
    hbond_count: int
    hbond_norm: float
    semantic_affinity: float
    stability_proxy: float
    transition_role: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FourBlockReport:
    schema: str
    bases: List[DNABaseBlock]
    relations: List[BaseRelation]
    complement_axes: Dict[str, Dict[str, Any]]
    relation_matrix: Dict[str, Dict[str, float]]
    metrics: Dict[str, float]
    interpretation: str
    certification_boundary: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "bases": [b.to_dict() for b in self.bases],
            "relations": [r.to_dict() for r in self.relations],
            "complement_axes": self.complement_axes,
            "relation_matrix": self.relation_matrix,
            "metrics": self.metrics,
            "interpretation": self.interpretation,
            "certification_boundary": self.certification_boundary,
        }

    def to_noema_candidate(self) -> Dict[str, Any]:
        return {
            "schema": "NOEMA_INGEST_CANDIDATE_V0_1",
            "namespace": "Origins-Of-Life",
            "project_id": "Origins-Of-Life::dna_four_blocks_relation",
            "layer": "episodic",
            "snap_type": "iteration_state",
            "tags": ["origins", "dna", "four_blocks", "semantic_compression", "base_pairing"],
            "meaning_candidate": {
                "hypothesis": (
                    "The four DNA blocks form two complementary axes, A-T and C-G, constrained by "
                    "purine/pyrimidine opposition and a stability gradient; the alphabet is therefore "
                    "a compact relational grammar rather than four independent symbols."
                ),
                "confidence": 0.82,
                "source": "local NOEMA CURRENT + deterministic Origins v0.5 DNA four-block relation model",
                "context_required": True,
                "canon_allowed": False,
                "needs_repair": False,
                "uncertainty_operator": "chyba",
            },
            "report": self.to_dict(),
        }


def _relation_kind(a: str, b: str) -> str:
    if a == b:
        return "identity_self"
    if COMPLEMENT[a] == b:
        return "watson_crick_complement"
    if BASE_CLASS[a] == BASE_CLASS[b]:
        return f"same_class_{BASE_CLASS[a]}"
    return "cross_class_noncomplement"


def relation_between(a: str, b: str) -> BaseRelation:
    a = a.upper()
    b = b.upper()
    if a not in BASE_ORDER or b not in BASE_ORDER:
        raise ValueError(f"DNA bases must be one of {BASE_ORDER}; got {a!r}, {b!r}")
    pair = frozenset((a, b))
    is_self = a == b
    is_complement = COMPLEMENT[a] == b
    h = PAIR_H_BONDS.get(pair, 0) if is_complement else 0
    hnorm = h / 3.0 if h else 0.0
    class_opposition = 1.0 if BASE_CLASS[a] != BASE_CLASS[b] else 0.0
    not_self = 0.0 if is_self else 1.0
    # Relation score: complement is primary; H-bond stiffness, class opposition,
    # typed duality and non-self transition add smaller contributions.
    semantic_affinity = _clamp01(
        0.50 * float(is_complement)
        + 0.20 * hnorm
        + 0.15 * class_opposition
        + 0.10 * (class_opposition * not_self)
        + 0.05 * not_self
    )
    # Stability proxy is stricter: non-complement cross-class contact is relation-
    # possible but not a stable Watson-Crick axis in this model.
    stability_proxy = _clamp01(0.70 * float(is_complement) + 0.30 * hnorm)
    if is_complement:
        role = PAIR_AXIS[pair]
    elif is_self:
        role = "self_identity_diagonal"
    elif BASE_CLASS[a] == BASE_CLASS[b]:
        role = "same_class_lateral_transition"
    else:
        role = "cross_class_mismatch_transition"
    return BaseRelation(
        source=a,
        target=b,
        relation_kind=_relation_kind(a, b),
        complementarity=float(is_complement),
        purine_pyrimidine_opposition=class_opposition,
        hbond_count=h,
        hbond_norm=hnorm,
        semantic_affinity=semantic_affinity,
        stability_proxy=stability_proxy,
        transition_role=role,
    )


def compute_four_block_relation() -> FourBlockReport:
    bases = [
        DNABaseBlock(
            symbol=b,
            chemical_class=BASE_CLASS[b],
            ring_count=RING_COUNT[b],
            complement=COMPLEMENT[b],
            semantic_role=("information_splitter" if b in {"A", "T"} else "stability_clamp"),
        )
        for b in BASE_ORDER
    ]
    relations = [relation_between(a, b) for a, b in product(BASE_ORDER, BASE_ORDER)]
    matrix: Dict[str, Dict[str, float]] = {a: {} for a in BASE_ORDER}
    for r in relations:
        matrix[r.source][r.target] = round(r.semantic_affinity, 6)

    comp = [r for r in relations if r.complementarity == 1.0]
    # Ordered relations include A->T and T->A, C->G and G->C. Axis metrics are
    # therefore averaged by unordered pair.
    at = [r for r in comp if frozenset((r.source, r.target)) == frozenset(("A", "T"))]
    cg = [r for r in comp if frozenset((r.source, r.target)) == frozenset(("C", "G"))]
    all_aff = [r.semantic_affinity for r in relations]
    offdiag = [r.semantic_affinity for r in relations if r.source != r.target]
    complement_aff = [r.semantic_affinity for r in comp]
    mismatch_aff = [r.semantic_affinity for r in relations if r.source != r.target and r.complementarity == 0.0]
    same_class = [r.semantic_affinity for r in relations if r.source != r.target and BASE_CLASS[r.source] == BASE_CLASS[r.target]]
    cross_class = [r.semantic_affinity for r in relations if r.source != r.target and BASE_CLASS[r.source] != BASE_CLASS[r.target]]

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    complement_selectivity = _clamp01(avg(complement_aff) - avg(mismatch_aff))
    axis_balance = _clamp01(1.0 - abs(avg([r.semantic_affinity for r in at]) - avg([r.semantic_affinity for r in cg])))
    alphabet_closure = _clamp01(len({r.source for r in comp} | {r.target for r in comp}) / 4.0)
    class_symmetry = _clamp01(1.0 - abs(len([b for b in BASE_ORDER if BASE_CLASS[b] == "purine"]) - len([b for b in BASE_ORDER if BASE_CLASS[b] == "pyrimidine"])) / 4.0)
    grammar_compression = _clamp01(1.0 - 2.0 / 16.0)  # two axes generate the 16 ordered relations as invariant grammar
    transition_readiness = _clamp01(
        0.30 * complement_selectivity
        + 0.22 * alphabet_closure
        + 0.18 * class_symmetry
        + 0.16 * axis_balance
        + 0.14 * grammar_compression
    )
    identity_retention = _clamp01(
        0.34 * complement_selectivity
        + 0.26 * axis_balance
        + 0.20 * alphabet_closure
        + 0.20 * class_symmetry
    )

    complement_axes = {
        "AT": {
            "bases": ["A", "T"],
            "hydrogen_bonds_proxy": 2,
            "stability_role": "lower_stability_labile_information_axis",
            "mean_semantic_affinity": round(avg([r.semantic_affinity for r in at]), 6),
            "mean_stability_proxy": round(avg([r.stability_proxy for r in at]), 6),
        },
        "CG": {
            "bases": ["C", "G"],
            "hydrogen_bonds_proxy": 3,
            "stability_role": "higher_stability_clamp_axis",
            "mean_semantic_affinity": round(avg([r.semantic_affinity for r in cg]), 6),
            "mean_stability_proxy": round(avg([r.stability_proxy for r in cg]), 6),
        },
    }
    metrics = {
        "ordered_relation_count": 16.0,
        "complement_ordered_relation_count": float(len(comp)),
        "unordered_complement_axis_count": 2.0,
        "mean_semantic_affinity_all": round(avg(all_aff), 6),
        "mean_semantic_affinity_offdiag": round(avg(offdiag), 6),
        "mean_complement_affinity": round(avg(complement_aff), 6),
        "mean_mismatch_affinity": round(avg(mismatch_aff), 6),
        "mean_same_class_affinity": round(avg(same_class), 6),
        "mean_cross_class_affinity": round(avg(cross_class), 6),
        "complement_selectivity_proxy": round(complement_selectivity, 6),
        "axis_balance_proxy": round(axis_balance, 6),
        "alphabet_closure_proxy": round(alphabet_closure, 6),
        "class_symmetry_proxy": round(class_symmetry, 6),
        "grammar_compression_proxy": round(grammar_compression, 6),
        "identity_retention_proxy": round(identity_retention, 6),
        "transition_readiness_proxy": round(transition_readiness, 6),
    }
    interpretation = (
        "The four DNA bases are not modeled as four independent blocks. They form two complementary "
        "axes, A-T and C-G. The A-T axis is the more labile two-bond information splitter; the C-G "
        "axis is the stiffer three-bond clamp. Purine/pyrimidine opposition closes the alphabet, and "
        "semantic compression preserves the relation grammar as two axes plus class symmetry."
    )
    return FourBlockReport(
        schema=SCHEMA,
        bases=bases,
        relations=relations,
        complement_axes=complement_axes,
        relation_matrix=matrix,
        metrics=metrics,
        interpretation=interpretation,
        certification_boundary={
            "empirical_abiogenesis_proof_claimed": False,
            "quantum_chemistry_claimed": False,
            "thermodynamic_dna_folding_claimed": False,
            "semantic_relation_model_claimed": True,
            "carbon_organic_scope": True,
        },
    )


def run_dna_four_block_demo(as_noema: bool = False) -> Mapping[str, Any] | FourBlockReport:
    report = compute_four_block_relation()
    return report.to_noema_candidate() if as_noema else report


__all__ = [
    "DNABaseBlock",
    "BaseRelation",
    "FourBlockReport",
    "relation_between",
    "compute_four_block_relation",
    "run_dna_four_block_demo",
]
