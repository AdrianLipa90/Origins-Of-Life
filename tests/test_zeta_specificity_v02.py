import numpy as np
import pytest

from origins.analysis.ablation import initial_state_digest
from origins.analysis.zeta_specificity_v02 import (
    PERMUTATION_SEED,
    SHIFT_CYCLES_PER_SAMPLE,
    V02_MODES,
    ZetaSpecificityV02Modulator,
    _base_targets,
    _hermitian_classes,
    _narrow_mask,
    _narrow_sigma,
    _simulator,
    hermitian_histogram_permutation,
    run_zeta_specificity_v02,
)
from origins.constants import RIEMANN_CRITICAL_ZEROS
from origins.scenarios import SCENARIO_A


def test_histogram_permutation_is_exact_and_hermitian():
    targets = _base_targets(RIEMANN_CRITICAL_ZEROS)
    base = _narrow_mask((32, 32), targets, _narrow_sigma(targets))
    permuted = hermitian_histogram_permutation(
        base,
        seed=PERMUTATION_SEED,
    )

    assert permuted[0, 0] == base[0, 0] == 1.0
    assert np.array_equal(
        np.sort(base.ravel()[1:]),
        np.sort(permuted.ravel()[1:]),
    )
    assert not np.array_equal(base, permuted)

    Nx, Ny = base.shape
    for i in range(Nx):
        for j in range(Ny):
            assert permuted[i, j] == permuted[(-i) % Nx, (-j) % Ny]


def test_frozen_target_placebos_preserve_declared_invariants():
    zeta = ZetaSpecificityV02Modulator(
        mode="narrow_zeta", sigma_heis=0.0
    ).target_set()
    reflected = ZetaSpecificityV02Modulator(
        mode="narrow_reflected", sigma_heis=0.0
    ).target_set()
    shifted = ZetaSpecificityV02Modulator(
        mode="narrow_shifted", sigma_heis=0.0
    ).target_set()
    equal = ZetaSpecificityV02Modulator(
        mode="narrow_equispaced", sigma_heis=0.0
    ).target_set()

    assert zeta is not None
    assert reflected is not None
    assert shifted is not None
    assert equal is not None

    assert np.mean(reflected) == pytest.approx(np.mean(zeta))
    assert np.ptp(reflected) == pytest.approx(np.ptp(zeta))
    assert np.allclose(
        np.sort(np.abs(zeta[:, None] - zeta[None, :]).ravel()),
        np.sort(np.abs(reflected[:, None] - reflected[None, :]).ravel()),
    )

    assert np.allclose(shifted - zeta, SHIFT_CYCLES_PER_SAMPLE)
    assert np.allclose(np.diff(shifted), np.diff(zeta))

    assert equal[0] == pytest.approx(np.min(zeta))
    assert equal[-1] == pytest.approx(np.max(zeta))
    assert np.allclose(np.diff(equal), np.diff(equal)[0])


def test_histogram_placebo_metadata_matches_zeta_histogram_exactly():
    zeta = ZetaSpecificityV02Modulator(
        mode="narrow_zeta", sigma_heis=0.0
    ).mask_diagnostics((32, 32))
    perm = ZetaSpecificityV02Modulator(
        mode="histogram_permuted", sigma_heis=0.0
    ).mask_diagnostics((32, 32))

    assert perm["histogram_exactly_matches_narrow_zeta"] is True
    assert perm["sorted_non_dc_mask_sha256"] == zeta["sorted_non_dc_mask_sha256"]
    assert perm["mask_sha256"] != zeta["mask_sha256"]
    assert perm["correlation_with_narrow_zeta"] < 0.99
    assert perm["mae_vs_narrow_zeta"] > 0.0


def test_v02_initial_state_is_matched_across_all_modes():
    digests = []
    for mode in V02_MODES:
        sim = _simulator(
            SCENARIO_A,
            mode,
            201,
            Nx=8,
            Ny=8,
            dt_h=0.05,
            outdir="unused-test-output",
        )
        sim.initialize()
        digests.append(initial_state_digest(sim))

    assert len(set(digests)) == 1


def test_v02_runner_rejects_post_prereg_parameter_drift():
    with pytest.raises(ValueError):
        run_zeta_specificity_v02(
            SCENARIO_A,
            seeds=[101],
            hours=120.0,
        )
    with pytest.raises(ValueError):
        run_zeta_specificity_v02(
            SCENARIO_A,
            seeds=[201],
            hours=12.0,
        )
