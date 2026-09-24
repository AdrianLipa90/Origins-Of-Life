import numpy as np

from origins.analysis.zeta_specificity import (
    ControlledZetaModulator,
    MODES,
    run_zeta_specificity_controls,
)
from origins.scenarios import SCENARIO_A


def _pairwise_distances(values):
    v = np.sort(np.asarray(values, dtype=float))
    return np.sort(np.abs(v[:, None] - v[None, :]).ravel())


def test_reflected_placebo_preserves_target_geometry():
    zeta = ControlledZetaModulator(
        mode="narrow_zeta", sigma_heis=0.0
    ).mapped_targets()
    placebo = ControlledZetaModulator(
        mode="narrow_reflected_placebo", sigma_heis=0.0
    ).mapped_targets()

    assert len(zeta) == len(placebo)
    assert np.mean(zeta) == np.mean(placebo)
    assert np.ptp(zeta) == np.ptp(placebo)
    assert np.allclose(_pairwise_distances(zeta), _pairwise_distances(placebo))
    assert not np.allclose(np.sort(zeta), np.sort(placebo))


def test_narrow_control_is_not_the_legacy_broad_collapse():
    narrow = ControlledZetaModulator(
        mode="narrow_zeta", lambda_soft=6.0, sigma_heis=0.0
    ).control_metadata((32, 32))
    broad = ControlledZetaModulator(
        mode="legacy_broad", lambda_soft=6.0, sigma_heis=0.0
    ).control_metadata((32, 32))

    assert narrow["non_dc_mask_mean"] > 0.95
    assert broad["non_dc_mask_mean"] < 0.01
    assert narrow["non_dc_fraction_below_0_9"] < 0.1


def test_specificity_runner_is_matched_across_modes():
    runs, contrasts = run_zeta_specificity_controls(
        SCENARIO_A,
        seeds=[1901],
        geometry_enabled=True,
        modes=MODES,
        Nx=8,
        Ny=8,
        dt_h=0.05,
        hours=0.10,
        include_clay=False,
        preseed_rna=True,
    )

    assert set(runs["mode"]) == set(MODES)
    assert runs["initial_state_digest"].nunique() == 1
    assert set(contrasts["mode"]) == set(MODES)
