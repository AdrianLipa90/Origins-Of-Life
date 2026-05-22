from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class OrbitalCoordinate:
    canonical_id: str
    radius: float
    theta: float
    phi: float
    omega: float
    tau_local: float
    semantic_mass: float
    attractor_charge: float
    coherence: float
    defect: float
    seed_norm: float
    orbit_index: int
    relation_depth: int
    sphere_id: Optional[str] = None
    parent_sphere_id: Optional[str] = None
    leak_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class OrbitalSystemState:
    coordinates: Dict[str, OrbitalCoordinate] = field(default_factory=dict)
    memory_state: Dict[str, object] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def add(self, coordinate: OrbitalCoordinate) -> None:
        self.coordinates[coordinate.canonical_id] = coordinate

    def to_dict(self) -> Dict[str, object]:
        return {
            "coordinates": {k: v.to_dict() for k, v in self.coordinates.items()},
            "memory_state": dict(self.memory_state),
            "metadata": dict(self.metadata),
        }
