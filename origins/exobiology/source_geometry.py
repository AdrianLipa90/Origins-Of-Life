from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .ammonia_runtime import AmmoniaCandidateSimulator
from .hydrocarbon_runtime import HydrocarbonCandidateSimulator


SCHEMA = "ORIGINS_EXOTIC_SOURCE_GEOMETRY_V0_1"


@dataclass(frozen=True)
class SourceGeometryDiagnostic:
    schema: str
    runtime: str
    profile: str
    source_information_norm: float
    source_boundary_norm: float
    cosine_overlap: float
    pearson_correlation: float
    proportionality_residual_relative: float | None
    exact_initial_proportionality_expected: bool
    physical_binding: str
    interpretation: str

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _cosine_overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.ravel(np.asarray(a, dtype=float))
    bb = np.ravel(np.asarray(b, dtype=float))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 0.0 else 0.0


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.ravel(np.asarray(a, dtype=float))
    bb = np.ravel(np.asarray(b, dtype=float))
    if aa.size != bb.size or aa.size == 0:
        raise ValueError("source fields must be same nonzero size")
    sa = float(np.std(aa))
    sb = float(np.std(bb))
    if sa <= 1e-15 or sb <= 1e-15:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def source_fields(simulator) -> tuple[np.ndarray, np.ndarray]:
    """Return the runtime's native instantaneous assembly-source geometry."""
    simulator._require_initialized()
    if not hasattr(simulator, "candidate_assembly_sources"):
        raise TypeError(f"unsupported exotic runtime: {type(simulator)!r}")
    sources = simulator.candidate_assembly_sources()
    information = np.asarray(sources["information"], dtype=float)
    boundary = np.asarray(sources["boundary"], dtype=float)
    if information.shape != boundary.shape or information.ndim != 2:
        raise ValueError("candidate assembly source fields must be same-shape 2D arrays")
    return information, boundary


def diagnose_source_geometry(simulator) -> SourceGeometryDiagnostic:
    source_information, source_boundary = source_fields(simulator)
    claim = simulator.claim_status()

    proportionality_residual_relative: float | None = None
    exact_expected = isinstance(simulator, AmmoniaCandidateSimulator) and bool(
        np.all(simulator.Q == 0.0)
    )
    if exact_expected and simulator.parameters.boundary_assembly_rate > 0.0:
        ratio = (
            simulator.parameters.information_assembly_rate
            / simulator.parameters.boundary_assembly_rate
        )
        residual = source_information - ratio * source_boundary
        denom = max(float(np.linalg.norm(source_information)), 1e-15)
        proportionality_residual_relative = float(np.linalg.norm(residual) / denom)

    cosine = _cosine_overlap(source_information, source_boundary)
    pearson = _pearson(source_information, source_boundary)

    if exact_expected:
        interpretation = "INITIAL_SOURCE_COLOCATION_EXACT_UP_TO_FLOATING_POINT"
    elif cosine > 0.0:
        interpretation = "SOURCE_OVERLAP_MEASURED_NO_CAUSAL_CLAIM"
    else:
        interpretation = "NO_POSITIVE_SOURCE_OVERLAP_DETECTED"

    return SourceGeometryDiagnostic(
        schema=SCHEMA,
        runtime=str(claim["runtime"]),
        profile=str(claim["profile"]),
        source_information_norm=float(np.linalg.norm(source_information)),
        source_boundary_norm=float(np.linalg.norm(source_boundary)),
        cosine_overlap=cosine,
        pearson_correlation=pearson,
        proportionality_residual_relative=proportionality_residual_relative,
        exact_initial_proportionality_expected=exact_expected,
        physical_binding=str(claim["physical_binding"]),
        interpretation=interpretation,
    )
