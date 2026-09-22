from __future__ import annotations

from .ammonia_runtime import AmmoniaCandidateSimulator
from .profiles import AMMONIA_CANDIDATE, HYDROCARBON_CANDIDATE


def create_exotic_candidate_simulator(config, **kwargs):
    """Create a dedicated exotic candidate runtime when one exists.

    The terracentric UniversalOriginSimulator is intentionally not returned
    here. Unsupported candidate profiles fail closed.
    """
    profile = getattr(config, "biochemistry_profile", None)
    if profile == AMMONIA_CANDIDATE.code:
        return AmmoniaCandidateSimulator(config, **kwargs)
    if profile == HYDROCARBON_CANDIDATE.code:
        raise NotImplementedError(
            "HYDROCARBON_CANDIDATE has no dedicated runtime yet"
        )
    raise ValueError(
        f"{profile!r} is not an exotic candidate profile with a dedicated runtime"
    )


def available_exotic_candidate_runtimes() -> tuple[str, ...]:
    return ("AMMONIA_CANDIDATE_V0_1",)
