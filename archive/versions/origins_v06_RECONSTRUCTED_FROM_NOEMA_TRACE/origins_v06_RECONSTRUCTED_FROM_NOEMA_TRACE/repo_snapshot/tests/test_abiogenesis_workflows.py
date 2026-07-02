from origins.abiogenesis.workflows import (
    default_feasibility_protocol,
    default_habitat_protocol,
    default_origin_comparison_protocol,
    default_single_origin_protocol,
)


def test_workflow_presets_are_exported_and_callable():
    assert callable(default_single_origin_protocol)
    assert callable(default_habitat_protocol)
    assert callable(default_feasibility_protocol)
    assert callable(default_origin_comparison_protocol)
