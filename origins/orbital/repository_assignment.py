from __future__ import annotations

from typing import Iterable

from ..bindings import scenario_config_to_entity_record
from ..registry import EntityRecord
from ..scenarios import ALL_SCENARIOS
from .state import OrbitalCoordinate, OrbitalSystemState
from .subjective_time import compute_local_subjective_time


def assign_orbital_state_to_entity(record: EntityRecord, delta_t: float = 1.0) -> OrbitalCoordinate:
    semantic_mass = max(1.0, float(getattr(record, "semantic_mass", 1.0)))
    relation_depth = max(1, int(record.relation_depth))
    radius = float(record.orbit_index + relation_depth)

    # Bloch sphere geometry (Fubini-Study):
    #   phase φ ∈ [0, 2π] — azimuthal angle on S²
    #   bloch_theta = φ / π  maps phase to polar angle ∈ [0, 2]
    #   coherence = cos²(θ/2) — overlap with |0⟩ (north pole = full coherence)
    import math
    phi = float(record.phase)
    bloch_theta = (abs(phi) % math.pi)  # polar angle ∈ [0, π]
    coherence = math.cos(bloch_theta / 2.0) ** 2  # ∈ [0, 1]
    defect = math.sin(bloch_theta / 2.0) ** 2     # = 1 - coherence (Euler: coh+defect=1)

    tau_local = compute_local_subjective_time(delta_t, radius, semantic_mass, coherence, defect)
    return OrbitalCoordinate(
        canonical_id=record.canonical_id,
        radius=radius,
        theta=bloch_theta,
        phi=phi,
        omega=0.0,
        tau_local=tau_local,
        semantic_mass=semantic_mass,
        attractor_charge=1.0,
        coherence=coherence,
        defect=defect,
        seed_norm=1.0,
        orbit_index=int(record.orbit_index),
        relation_depth=relation_depth,
        sphere_id=getattr(record, "sphere_id", None),
        parent_sphere_id=getattr(record, "parent_sphere_id", None),
        leak_mode=getattr(record, "leak_mode", None),
    )



def bind_live_topology_to_coordinate(
    coordinate: OrbitalCoordinate,
    topology,
    *,
    delta_t: float,
) -> OrbitalCoordinate:
    """Bind live candidate-geometry coherence before downstream calculations.

    The operation is deterministic and keeps coherence/defect/tau_local
    internally consistent.  It does not promote the geometry to a physical
    mechanism.
    """
    if topology is None or not hasattr(topology, "bloch_coherence"):
        return coordinate
    coherence = float(topology.bloch_coherence())
    if not 0.0 <= coherence <= 1.0:
        raise ValueError("live topology coherence must lie in [0, 1]")
    coordinate.coherence = coherence
    coordinate.defect = 1.0 - coherence
    coordinate.tau_local = compute_local_subjective_time(
        delta_t,
        coordinate.radius,
        coordinate.semantic_mass,
        coordinate.coherence,
        coordinate.defect,
    )
    return coordinate


def build_repository_system_state(configs: Iterable = ALL_SCENARIOS) -> OrbitalSystemState:
    state = OrbitalSystemState(metadata={"hierarchy": ["relation", "identity", "memory", "process", "artifact"]})
    for cfg in configs:
        rec = scenario_config_to_entity_record(cfg)
        state.add(assign_orbital_state_to_entity(rec))
    return state
