from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SemanticLayer(str, Enum):
    RELATION = "relation"
    IDENTITY = "identity"
    MEMORY = "memory"
    PROCESS = "process"
    ARTIFACT = "artifact"


class ObjectType(str, Enum):
    PACKAGE = "package"
    MODULE = "module"
    SCENARIO = "scenario"
    SIMULATOR = "simulator"
    ANALYSIS = "analysis"
    REPORT = "report"
    REGISTRY = "registry"
    DOCUMENT = "document"
    SCRIPT = "script"
    TEST = "test"


class Sector(str, Enum):
    CORE = "core"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    TOPOLOGY = "topology"
    COSMOLOGY = "cosmology"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    DOCS = "docs"
    TESTS = "tests"
    SCRIPTS = "scripts"


class EpistemicStatus(str, Enum):
    CANONICAL = "canonical"
    WORKING = "working"
    DERIVED = "derived"
    EXPERIMENTAL = "experimental"
    LEGACY = "legacy"


class AttractorId(str, Enum):
    ATTRACTOR_RUNTIME = "AttractorRuntime"
    ATTRACTOR_TOPOLOGY = "AttractorTopology"
    ATTRACTOR_LLM_TEMP = "AttractorLLMTemp"


@dataclass(frozen=True)
class DefinitionalContract:
    semantic_hierarchy: List[SemanticLayer] = field(
        default_factory=lambda: [
            SemanticLayer.RELATION,
            SemanticLayer.IDENTITY,
            SemanticLayer.MEMORY,
            SemanticLayer.PROCESS,
            SemanticLayer.ARTIFACT,
        ]
    )
    separation_of_description_levels: List[str] = field(
        default_factory=lambda: [
            "state_formalism",
            "computational_implementation",
            "visual_geometry",
            "native_renderer",
        ]
    )
    llm_role: str = "temporary_attractor"


DEFAULT_DEFINITIONAL_CONTRACT = DefinitionalContract()


@dataclass
class AttractorWeights:
    weights: Dict[AttractorId, float]

    def normalized(self) -> "AttractorWeights":
        total = sum(self.weights.values())
        if total <= 0:
            return AttractorWeights(weights=dict(self.weights))
        return AttractorWeights(weights={k: v / total for k, v in self.weights.items()})


@dataclass
class ScenarioDefinition:
    canonical_id: str
    name: str
    code: str
    object_type: ObjectType = ObjectType.SCENARIO
    sector: Sector = Sector.CHEMISTRY
    semantic_layer: SemanticLayer = SemanticLayer.PROCESS
    epistemic_status: EpistemicStatus = EpistemicStatus.WORKING
    orbit_index: int = 0
    phase: float = 0.0
    winding_number: int = 0
    relation_depth: int = 1
    semantic_mass: float = 1.0
    subjective_time_scale: float = 1.0
    sphere_id: Optional[str] = None
    parent_sphere_id: Optional[str] = None
    provenance_links: List[str] = field(default_factory=list)
    dependency_links: List[str] = field(default_factory=list)
    attractor_weights: Optional[AttractorWeights] = None
    leak_mode: Optional[str] = None
