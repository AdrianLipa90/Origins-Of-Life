"""Integration tests for UniversalOriginSimulator."""
import unittest
import numpy as np
from origins.scenarios import SCENARIO_A, SCENARIO_C
from origins.simulator import UniversalOriginSimulator


class TestSimulatorSmoke(unittest.TestCase):
    """Quick smoke tests — run a handful of steps to catch import/shape errors."""

    def _make_sim(self, cfg=SCENARIO_A, Nx=16, Ny=16):
        sim = UniversalOriginSimulator(cfg, Nx=Nx, Ny=Ny, dt_h=0.1, outdir="/tmp/test_sim")
        sim.initialize()
        return sim

    def test_initialize_fields_not_none(self):
        sim = self._make_sim()
        for attr in ('E', 'O', 'N', 'R', 'M', 'L', 'Cat'):
            self.assertIsNotNone(getattr(sim, attr))

    def test_fields_correct_shape(self):
        sim = self._make_sim(Nx=16, Ny=16)
        for attr in ('E', 'O', 'N', 'R', 'M', 'L', 'Cat'):
            self.assertEqual(getattr(sim, attr).shape, (16, 16))

    def test_step_does_not_crash(self):
        sim = self._make_sim()
        sim.step()

    def test_run_returns_dataframe(self):
        import pandas as pd
        sim = self._make_sim()
        df = sim.run(hours=0.5, record_interval=0.2, verbose=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('time_h', df.columns)
        self.assertIn('n_protocells', df.columns)

    def test_history_grows(self):
        sim = self._make_sim()
        sim.run(hours=1.0, record_interval=0.2, verbose=False)
        self.assertGreater(len(sim.history['time_h']), 0)

    def test_fields_non_negative_after_run(self):
        sim = self._make_sim()
        sim.run(hours=1.0, verbose=False)
        for attr in ('E', 'O', 'N', 'R', 'M', 'L'):
            field = getattr(sim, attr)
            self.assertGreaterEqual(field.min(), 0.0, msg=f"{attr} has negative values")

    def test_ammonia_scenario(self):
        """Ammonia scenario (C) should run without errors."""
        sim = self._make_sim(cfg=SCENARIO_C)
        df = sim.run(hours=1.0, verbose=False)
        self.assertIsNotNone(df)


if __name__ == "__main__":
    unittest.main()
