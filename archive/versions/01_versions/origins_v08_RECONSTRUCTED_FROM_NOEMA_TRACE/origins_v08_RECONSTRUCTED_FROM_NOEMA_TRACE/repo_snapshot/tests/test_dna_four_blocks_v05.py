from origins.abiogenesis.dna_four_blocks import compute_four_block_relation, relation_between


def test_four_block_report_axes_and_metrics():
    report = compute_four_block_relation()
    data = report.to_dict()
    assert data["schema"] == "ORIGINS_NOEMA_DNA_FOUR_BLOCK_RELATION_V0_5"
    assert data["complement_axes"]["AT"]["hydrogen_bonds_proxy"] == 2
    assert data["complement_axes"]["CG"]["hydrogen_bonds_proxy"] == 3
    assert data["metrics"]["ordered_relation_count"] == 16.0
    assert data["metrics"]["unordered_complement_axis_count"] == 2.0
    assert data["metrics"]["alphabet_closure_proxy"] == 1.0
    assert data["metrics"]["transition_readiness_proxy"] > 0.75


def test_complement_edges_score_above_mismatch_edges():
    at = relation_between("A", "T")
    cg = relation_between("C", "G")
    ac = relation_between("A", "C")
    ag = relation_between("A", "G")
    assert at.relation_kind == "watson_crick_complement"
    assert cg.stability_proxy > at.stability_proxy
    assert at.semantic_affinity > ac.semantic_affinity > ag.semantic_affinity
    assert cg.semantic_affinity >= at.semantic_affinity


def test_noema_candidate_boundary_is_honest():
    candidate = compute_four_block_relation().to_noema_candidate()
    boundary = candidate["report"]["certification_boundary"]
    assert boundary["semantic_relation_model_claimed"] is True
    assert boundary["empirical_abiogenesis_proof_claimed"] is False
    assert boundary["quantum_chemistry_claimed"] is False
