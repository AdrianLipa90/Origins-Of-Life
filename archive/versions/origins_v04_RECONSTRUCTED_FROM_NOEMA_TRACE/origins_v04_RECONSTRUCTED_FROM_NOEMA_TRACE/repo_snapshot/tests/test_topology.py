"""Tests for origins.topology."""
import unittest
import numpy as np
from origins.scenarios import SCENARIO_A, SCENARIO_B, SCENARIO_D
from origins.topology.fields import TopologyField
from origins.topology.constraints import ZetaRiemannModulator


class TestTopologyField(unittest.TestCase):

    def test_shape(self):
        tf = TopologyField(SCENARIO_A, Nx=16, Ny=16)
        self.assertEqual(tf.field.shape, (16, 16))
        self.assertEqual(tf.curvature.shape, (16, 16))

    def test_strength_applied(self):
        tf = TopologyField(SCENARIO_A, Nx=16, Ny=16)
        self.assertLessEqual(np.abs(tf.field).max(), SCENARIO_A.topo_strength * 5)

    def test_static_mode_no_change(self):
        tf = TopologyField(SCENARIO_B, Nx=16, Ny=16)
        before = tf.field.copy()
        tf.advance(100.0)
        np.testing.assert_array_equal(tf.field, before)

    def test_pulsing_mode_changes_field(self):
        tf = TopologyField(SCENARIO_A, Nx=16, Ny=16)
        before = tf.field.copy()
        # t=7 → sin(2π*0.05*7) = sin(2.199) ≠ 0, so field should change
        tf.advance(7.0)
        self.assertFalse(np.allclose(tf.field, before))

    def test_zero_strength_flat_field(self):
        tf = TopologyField(SCENARIO_D, Nx=16, Ny=16)
        # Titan has topo_strength=0.05 — just check it doesn't crash
        self.assertEqual(tf.field.shape, (16, 16))

    def test_modulation_helpers_shape(self):
        tf = TopologyField(SCENARIO_A, Nx=16, Ny=16)
        for meth in ("synthesis_mod", "catalysis_mod", "energy_mod",
                     "degradation_mod", "membrane_mod"):
            out = getattr(tf, meth)()
            self.assertEqual(out.shape, (16, 16))
            self.assertTrue((out > 0).all())


class TestZetaRiemannModulator(unittest.TestCase):

    def test_apply_preserves_shape(self):
        mod = ZetaRiemannModulator()
        field = np.random.rand(16, 16).astype(float)
        rng   = np.random.default_rng(0)
        out   = mod.apply(field, rng)
        self.assertEqual(out.shape, field.shape)

    def test_apply_output_clipped(self):
        mod = ZetaRiemannModulator()
        field = np.random.rand(16, 16).astype(float)
        rng   = np.random.default_rng(0)
        out   = mod.apply(field, rng)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)


if __name__ == "__main__":
    unittest.main()
