"""Tests for origins.chemistry."""
import unittest
import numpy as np
from origins.chemistry.fields import (
    init_field, laplacian, diffuse_field,
    solar_envelope, tide_semidiurnal,
)
from origins.chemistry.clay import ClayMineral


class TestFieldUtils(unittest.TestCase):

    def test_init_field_shape(self):
        f = init_field(32, 32, mean=0.5)
        self.assertEqual(f.shape, (32, 32))

    def test_init_field_clipped(self):
        f = init_field(32, 32, mean=0.5)
        self.assertGreaterEqual(f.min(), 0.0)
        self.assertLessEqual(f.max(), 1.0)

    def test_laplacian_flat_is_zero(self):
        f = np.ones((16, 16), dtype=float)
        lap = laplacian(f)
        np.testing.assert_allclose(lap, 0.0, atol=1e-10)

    def test_laplacian_shape(self):
        f = np.random.rand(16, 16)
        self.assertEqual(laplacian(f).shape, (16, 16))

    def test_diffuse_field_non_negative(self):
        f = np.random.rand(16, 16).astype(np.float32)
        out = diffuse_field(f, D=0.1, dt=0.05)
        self.assertGreaterEqual(out.min(), 0.0)

    def test_solar_envelope_range(self):
        for t in range(0, 48):
            v = solar_envelope(float(t))
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-9)

    def test_tide_semidiurnal_range(self):
        for t in range(0, 25):
            v = tide_semidiurnal(float(t))
            self.assertGreaterEqual(v, -1.0 - 1e-9)
            self.assertLessEqual(v, 1.0 + 1e-9)


class TestClayMineral(unittest.TestCase):

    def setUp(self):
        self.clay = ClayMineral(position=(10, 10), conc_g_L=5.0)

    def test_adsorption_non_negative(self):
        adsorbed = self.clay.adsorb_nucleotides(1.0)
        self.assertGreaterEqual(adsorbed, 0.0)

    def test_adsorption_does_not_exceed_available(self):
        n_free = 0.01
        adsorbed = self.clay.adsorb_nucleotides(n_free)
        self.assertLessEqual(adsorbed, n_free + 1e-9)

    def test_polymerization_positive(self):
        result = self.clay.catalyze_polymerization(0.5)
        self.assertGreater(result, 0.0)

    def test_concentration_boost(self):
        self.assertGreater(self.clay.get_concentration_boost(), 1.0)


if __name__ == "__main__":
    unittest.main()
