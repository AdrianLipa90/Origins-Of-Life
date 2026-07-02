from origins import ALL_SCENARIOS, SCENARIOS_BY_CODE
from origins.simulator.universal import UniversalOriginSimulator

def test_scenarios_available():
    assert len(ALL_SCENARIOS) >= 5
    assert "A" in SCENARIOS_BY_CODE
    assert "E" in SCENARIOS_BY_CODE

def test_simulator_imports():
    assert UniversalOriginSimulator is not None
