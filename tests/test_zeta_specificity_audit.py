import pytest

from origins.topology.constraints import ZetaRiemannModulator


@pytest.mark.parametrize("shape", [(32, 32), (96, 96)])
def test_legacy_zeta_mask_specificity_is_explicitly_quarantined(shape):
    mod = ZetaRiemannModulator(lambda_soft=6.0, sigma_heis=0.0)
    diag = mod.spectral_specificity_diagnostics(shape)

    assert diag["target_count"] == 6
    assert diag["target_min"] == pytest.approx(0.4669633359101931)
    assert diag["target_max"] == pytest.approx(0.4870419994713136)
    assert diag["single_notch_half_power_width"] > 0.3

    # The implemented six-zero mask is effectively a collapsed single-band
    # control under the current width. This is a known-model limitation, not a
    # physical conclusion about zeta zeros.
    assert diag["collapsed_mean_target_control_correlation"] > 0.9999
    assert diag["zero_specificity_resolved"] is False


def test_32_grid_radial_shell_audit_does_not_use_axis_bin_heuristic():
    mod = ZetaRiemannModulator(lambda_soft=6.0, sigma_heis=0.0)
    diag = mod.spectral_specificity_diagnostics((32, 32))

    # k=sqrt(kx^2+ky^2) has denser radial shells than the axis spacing 1/N.
    # Five of six targets map to distinct nearest radial shells at N=32.
    assert diag["distinct_nearest_radial_shells"] == 5
    assert diag["non_dc_fraction_below_1e3"] > 0.8
