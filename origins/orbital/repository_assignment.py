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
    coherence = max(0.0, 1.0 - float(record.phase) % 1.0)
    defect = abs(float(record.phase))
    tau_local = compute_local_subjective_time(delta_t, radius, semantic_mass, coherence, defect)
    return OrbitalCoordinate(
        canonical_id=record.canonical_id,
        radius=radius,
        theta=float(record.orbit_index),
        phi=float(record.phase),
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


def build_repository_system_state(configs: Iterable = ALL_SCENARIOS) -> OrbitalSystemState:
    state = OrbitalSystemState(metadata={"hierarchy": ["relation", "identity", "memory", "process", "artifact"]})
    for cfg in configs:
        rec = scenario_config_to_entity_record(cfg)
        state.add(assign_orbital_state_to_entity(rec))
    return state
