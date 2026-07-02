"""Tests for origins.biology."""
import unittest
import numpy as np
from origins.biology.rna import RNASequence, RNAPopulation
from origins.biology.protocell import ProtocellDetector


class TestRNASequence(unittest.TestCase):

    def setUp(self):
        self.rna = RNASequence("AUGCAUGCAUGCAUGCAUGC", position=(5, 5))

    def test_fitness_range(self):
        self.assertGreaterEqual(self.rna.fitness, 0.0)
        self.assertLessEqual(self.rna.fitness, 1.0)

    def test_replicate_returns_rna(self):
        daughter = self.rna.replicate()
        self.assertIsInstance(daughter, RNASequence)

    def test_replicate_increments_generation(self):
        daughter = self.rna.replicate()
        self.assertEqual(daughter.generation, 1)

    def test_replicate_same_length(self):
        daughter = self.rna.replicate(fidelity=1.0)
        self.assertEqual(daughter.length, self.rna.length)


class TestRNAPopulation(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.pop = RNAPopulation.seed(10, Nx=32, Ny=32, rng=rng)

    def test_seed_size(self):
        self.assertEqual(self.pop.size, 10)

    def test_fitness_range(self):
        self.assertGreaterEqual(self.pop.fitness.min(), 0.0)
        self.assertLessEqual(self.pop.fitness.max(), 1.0)

    def test_degrade_reduces_size(self):
        rng = np.random.default_rng(42)
        R_field  = np.ones((32, 32)) * 0.5
        topo     = np.zeros((32, 32))
        curv     = np.zeros((32, 32))
        initial  = self.pop.size
        # Use high degradation to ensure some molecules die
        self.pop.degrade(k_deg=2.0, temp_factor=1.0, topo_field=topo,
                         topo_curvature=curv, dt=0.05, rng=rng)
        self.assertLessEqual(self.pop.size, initial)

    def test_replicate_increases_size(self):
        rng = np.random.default_rng(7)
        R_field = np.ones((32, 32))
        topo    = np.zeros((32, 32))
        curv    = np.zeros((32, 32))
        before  = self.pop.size
        self.pop.replicate_and_select(R_field, topo, curv, dt=1.0, rng=rng)
        self.assertGreaterEqual(self.pop.size, before)


class TestProtocellDetector(unittest.TestCase):

    def test_detect_no_protocells(self):
        det = ProtocellDetector()
        M = np.zeros((32, 32))
        R = np.zeros((32, 32))
        self.assertEqual(det.detect(M, R), 0)

    def test_detect_some_protocells(self):
        det = ProtocellDetector(threshold_M=0.05, threshold_R=0.03)
        M = np.full((32, 32), 0.1)
        R = np.full((32, 32), 0.1)
        self.assertGreater(det.detect(M, R), 0)

    def test_advanced_returns_dict(self):
        det = ProtocellDetector()
        M = np.full((32, 32), 0.1)
        R = np.full((32, 32), 0.1)
        result = det.detect_advanced(M, R)
        self.assertIn('count', result)
        self.assertIn('stability_mean', result)


if __name__ == "__main__":
    unittest.main()
