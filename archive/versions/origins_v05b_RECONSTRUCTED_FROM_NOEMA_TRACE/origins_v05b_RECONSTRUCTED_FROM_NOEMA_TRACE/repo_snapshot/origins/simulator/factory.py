from __future__ import annotations

from typing import Literal

from ..scenarios import ScenarioConfig
from .universal import UniversalOriginSimulator
from .universal_orbital import OrbitalUniversalOriginSimulator

SimulatorMode = Literal["standard", "orbital"]


def create_simulator(
    config: ScenarioConfig,
    mode: SimulatorMode = "standard",
    Nx: int = 96,
    Ny: int = 96,
    dt_h: float = 0.05,
    outdir: str = "outputs",
    include_clay: bool = True,
):
    if mode == "orbital":
        return OrbitalUniversalOriginSimulator(
            config,
            Nx=Nx,
            Ny=Ny,
            dt_h=dt_h,
            outdir=outdir,
            include_clay=include_clay,
        )
    return UniversalOriginSimulator(
        config,
        Nx=Nx,
        Ny=Ny,
        dt_h=dt_h,
        outdir=outdir,
        include_clay=include_clay,
    )


def available_simulator_modes() -> list[str]:
    return ["standard", "orbital"]
