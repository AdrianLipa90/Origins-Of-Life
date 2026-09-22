from copy import deepcopy

import pandas as pd

from origins.analysis.ablation import (
    ARM_ORDER,
    METRICS,
    factorial_effects,
    run_matched_factorial_ablation,
)
from origins.scenarios import SCENARIO_A
from origins.simulator import UniversalOriginSimulator


def test_zeta_operator_does_not_advance_core_rng_stream():
    cfg = deepcopy(SCENARIO_A)
    cfg.seed = 1701
    cfg.use_zeta_constraints = True

    sim = UniversalOriginSimulator(
        cfg, Nx=8, Ny=8, include_clay=False, preseed_rna=True
    )
    sim.initialize()

    before = deepcopy(sim._rng.bit_generator.state)
    sim._apply_zeta_constraints()
    after = deepcopy(sim._rng.bit_generator.state)

    assert before == after


def test_matched_ablation_starts_from_identical_core_state():
    runs, effects = run_matched_factorial_ablation(
        SCENARIO_A,
        seeds=[1701],
        Nx=8,
        Ny=8,
        dt_h=0.05,
        hours=0.10,
        include_clay=False,
        preseed_rna=True,
    )

    assert set(runs["arm"]) == set(ARM_ORDER)
    assert runs["initial_state_digest"].nunique() == 1
    assert len(effects) == len(METRICS)
    assert set(effects["metric"]) == set(METRICS)


def test_factorial_effects_uses_standard_2x2_contrast():
    rows = []
    values = {
        "chemistry_only": 10.0,
        "zeta_only": 13.0,
        "geometry_only": 12.0,
        "full": 20.0,
    }
    for arm in ARM_ORDER:
        row = {"seed": 1, "arm": arm}
        row.update({metric: values[arm] for metric in METRICS})
        rows.append(row)

    effects = factorial_effects(pd.DataFrame(rows))
    first = effects.iloc[0]

    assert first["zeta_at_zero_geometry"] == 3.0
    assert first["geometry_at_zero_zeta"] == 2.0
    assert first["interaction"] == 5.0
    assert first["full_minus_chemistry"] == 10.0
