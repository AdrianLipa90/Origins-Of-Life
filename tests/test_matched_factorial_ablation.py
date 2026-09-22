from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pandas as pd

from origins.analysis.ablation import (
    ARM_ORDER,
    METRICS,
    factorial_effects,
    run_matched_factorial_ablation,
)
from origins.scenarios import SCENARIO_A
from origins.simulator import UniversalOriginSimulator
from origins.topology.constraints import ZetaRiemannModulator


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


def test_ablation_cli_is_directly_executable():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_matched_ablation.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Matched 2x2 ablation" in proc.stdout


def test_zeta_mask_preserves_dc_component():
    mod = ZetaRiemannModulator(lambda_soft=6.0, sigma_heis=0.001)
    mask = mod.spectral_mask((32, 32))
    assert mask[0, 0] == 1.0


def test_zeta_application_is_material_neutral():
    rng_field = __import__("numpy").random.default_rng(7001)
    field = rng_field.uniform(0.0, 1.0, (32, 32))
    before = float(field.sum())

    mod = ZetaRiemannModulator(lambda_soft=6.0, sigma_heis=0.001)
    out = mod.apply(field, __import__("numpy").random.default_rng(8001), 0.9)

    assert out.shape == field.shape
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()
    assert abs(float(out.sum()) - before) < 1e-8


def test_zeta_material_neutrality_handles_boundary_fields():
    import numpy as np

    mod = ZetaRiemannModulator(lambda_soft=6.0, sigma_heis=0.001)
    for value in (0.0, 1.0):
        field = np.full((8, 8), value)
        out = mod.apply(field, np.random.default_rng(9001), 0.9)
        assert abs(float(out.sum()) - float(field.sum())) < 1e-8
