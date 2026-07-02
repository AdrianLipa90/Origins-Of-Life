from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class SphereEmbedding:
    parent_sphere_id: Optional[str] = None
    child_sphere_ids: List[str] = field(default_factory=list)
    leak_mode: str = "bounded"
    chart_type: str = "bloch_local"

    def to_dict(self):
        return asdict(self)


@dataclass
class OrbitalSphere:
    sphere_id: str
    center_label: str
    radius: float
    chart_type: str = "bloch_local"
    embedding: SphereEmbedding = field(default_factory=SphereEmbedding)

    def contains_radius(self, r: float) -> bool:
        return r <= self.radius

    def to_dict(self):
        data = asdict(self)
        data["embedding"] = self.embedding.to_dict()
        return data
