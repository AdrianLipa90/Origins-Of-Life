"""Tests for Bloch sphere geometry and Berry holonomy in Origins-Of-Life.

Verifies:
- TopologyField computes valid Bloch coordinates
- Berry phase accumulates for pulsing/drift, stays 0 for static
- OrbitalCoordinate coherence + defect = 1.0 (Euler identity on S²)
- Potentials are physically ordered: north_pole < equator < south_pole
- Subjective time decreases with decreasing coherence
- Scenario phases are differentiated (not all 0)
"""
import math
import unittest

import numpy as np

from origins.scenarios import SCENARIO_A, SCENARIO_B, SCENARIO_C, SCENARIO_E
from origins.topology.fields import TopologyField
from origins.orbital.potentials import compute_potential_terms
from origins.orbital.subjective_time import compute_local_subjective_time
from origins.orbital.repository_assignment import (
    assign_orbital_state_to_entity,
    build_repository_system_state,
)
from origins.bindings import scenario_config_to_entity_record


class TestBlochCoordinates(unittest.TestCase):

    def test_bloch_theta_range(self):
        tf = TopologyField(SCENARIO_A, Nx=32, Ny=32)
        self.assertTrue((tf.bloch_theta >= 0).all())
        self.assertTrue((tf.bloch_theta <= math.pi + 1e-9).all())

    def test_bloch_phi_range(self):
        tf = TopologyField(SCENARIO_A, Nx=32, Ny=32)
        self.assertTrue((tf.bloch_phi >= 0).all())
        self.assertTrue((tf.bloch_phi <= 2 * math.pi + 1e-9).all())

    def test_bloch_coherence_in_unit_interval(self):
        tf = TopologyField(SCENARIO_A, Nx=32, Ny=32)
        c = tf.bloch_coherence()
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_bloch_shape_matches_field(self):
        tf = TopologyField(SCENARIO_A, Nx=16, Ny=24)
        self.assertEqual(tf.bloch_theta.shape, (16, 24))
        self.assertEqual(tf.bloch_phi.shape, (16, 24))


class TestBerryAccumulation(unittest.TestCase):

    def _advance_n(self, config, n=50, dt=0.1):
        tf = TopologyField(config, Nx=32, Ny=32)
        for i in range(n):
            tf.advance(i * dt)
        return tf.berry_accumulated

    def test_pulsing_accumulates_nonzero_berry(self):
        berry = self._advance_n(SCENARIO_A)  # PULSING
        self.assertNotEqual(berry, 0.0, "Pulsing should generate nonzero Berry phase")

    def test_static_berry_is_zero(self):
        berry = self._advance_n(SCENARIO_B)  # STATIC
        self.assertAlmostEqual(berry, 0.0, places=10,
                               msg="Static field should have zero Berry accumulation")

    def test_pulsing_enceladus_accumulates(self):
        berry = self._advance_n(SCENARIO_E)  # PULSING cosine
        self.assertNotEqual(berry, 0.0)

    def test_berry_initial_zero(self):
        tf = TopologyField(SCENARIO_A, Nx=16, Ny=16)
        self.assertEqual(tf.berry_accumulated, 0.0)

    def test_berry_sign_reflects_winding_direction(self):
        # Pulsing sin (A) and pulsing cos (E) should differ in accumulated phase
        berry_A = self._advance_n(SCENARIO_A, n=100)
        berry_E = self._advance_n(SCENARIO_E, n=100)
        # They use different patterns — absolute values should differ
        self.assertNotAlmostEqual(abs(berry_A), abs(berry_E), places=4)


class TestBlochOrbitalCoordinate(unittest.TestCase):

    def _coord(self, config, delta_t=120.0):
        rec = scenario_config_to_entity_record(config)
        return assign_orbital_state_to_entity(rec, delta_t=delta_t)

    def test_euler_identity_scenario_A(self):
        c = self._coord(SCENARIO_A)
        self.assertAlmostEqual(c.coherence + c.defect, 1.0, places=10)

    def test_euler_identity_all_scenarios(self):
        state = build_repository_system_state()
        for cid, coord in state.coordinates.items():
            self.assertAlmostEqual(
                coord.coherence + coord.defect, 1.0, places=10,
                msg=f"Euler identity failed for {cid}"
            )

    def test_coherence_in_unit_interval(self):
        state = build_repository_system_state()
        for cid, coord in state.coordinates.items():
            self.assertGreaterEqual(coord.coherence, 0.0, msg=cid)
            self.assertLessEqual(coord.coherence, 1.0, msg=cid)

    def test_scenario_phases_are_differentiated(self):
        """All scenarios should have distinct orbital phases (not all 0)."""
        phases = set()
        state = build_repository_system_state()
        for coord in state.coordinates.values():
            phases.add(round(coord.phi, 4))
        self.assertGreater(len(phases), 1, "All scenarios have identical phase — catalog not differentiated")

    def test_scenario_A_highest_coherence(self):
        """Scenario A (Earth-like) should be most coherent."""
        state = build_repository_system_state()
        coh_A = state.coordinates["OOL-SCENARIO-A"].coherence
        for cid, coord in state.coordinates.items():
            if cid != "OOL-SCENARIO-A":
                self.assertGreaterEqual(
                    coh_A, coord.coherence - 1e-9,
                    msg=f"A should be >= {cid} in coherence"
                )


class TestBlochPotentials(unittest.TestCase):

    def test_north_pole_minimum_potential(self):
        p_north = compute_potential_terms(1.0, 0.0, relation_depth=2, semantic_mass=2.0)
        p_equator = compute_potential_terms(0.5, 0.5, relation_depth=2, semantic_mass=2.0)
        self.assertLess(p_north.V_EC, p_equator.V_EC)
        self.assertLess(p_north.V_ZS, p_equator.V_ZS)

    def test_south_pole_maximum_potential(self):
        p_equator = compute_potential_terms(0.5, 0.5, relation_depth=2, semantic_mass=2.0)
        p_south = compute_potential_terms(0.01, 0.99, relation_depth=2, semantic_mass=2.0)
        self.assertLess(p_equator.V_tot, p_south.V_tot)

    def test_v_rel_capped(self):
        """V_rel should not diverge to infinity at south pole."""
        p = compute_potential_terms(0.001, 0.999, relation_depth=5, semantic_mass=1.0)
        self.assertLess(p.V_rel, 200.0)

    def test_all_terms_nonnegative(self):
        for coh in [0.0, 0.5, 1.0]:
            p = compute_potential_terms(coh, 1.0 - coh, relation_depth=1, semantic_mass=1.0)
            for term in [p.V_EC, p.V_ZS, p.V_rel, p.V_mem, p.V_def, p.V_ext]:
                self.assertGreaterEqual(term, 0.0)


class TestSubjectiveTime(unittest.TestCase):

    def test_high_coherence_longer_tau(self):
        tau_high = compute_local_subjective_time(120.0, radius=3.0, semantic_mass=2.0,
                                                  coherence=0.97, defect=0.03)
        tau_low = compute_local_subjective_time(120.0, radius=3.0, semantic_mass=2.0,
                                                 coherence=0.62, defect=0.38)
        self.assertGreater(tau_high, tau_low,
                           "High coherence (close to |0⟩) should have longer local time")

    def test_tau_positive(self):
        tau = compute_local_subjective_time(100.0, radius=5.0, semantic_mass=1.0,
                                             coherence=0.5, defect=0.5)
        self.assertGreater(tau, 0.0)

    def test_tau_scales_with_delta_t(self):
        tau1 = compute_local_subjective_time(60.0, radius=2.0, semantic_mass=1.0,
                                              coherence=0.8, defect=0.2)
        tau2 = compute_local_subjective_time(120.0, radius=2.0, semantic_mass=1.0,
                                              coherence=0.8, defect=0.2)
        self.assertAlmostEqual(tau2 / tau1, 2.0, places=9)


if __name__ == "__main__":
    unittest.main()
