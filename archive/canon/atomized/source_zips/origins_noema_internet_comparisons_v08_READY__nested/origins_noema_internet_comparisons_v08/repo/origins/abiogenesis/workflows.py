"""Canonical workflow presets for repo-facing abiogenesis usage.

These are not duplicate public APIs.
They are opinionated presets built on top of the canonical abiogenesis surface.
"""

from .api import (
    create_abiogenesis_runtime,
    run_feasibility_scan,
    run_habitat_scan,
    run_origin_comparison,
)


def default_single_origin_protocol(config, outdir: str = "outputs_abiogenesis", **kwargs):
    runtime = create_abiogenesis_runtime(
        config=config,
        mode="orbital",
        outdir=outdir,
        Nx=kwargs.pop("Nx", 96),
        Ny=kwargs.pop("Ny", 96),
        dt_h=kwargs.pop("dt_h", 0.05),
        **kwargs,
    )
    runtime.initialize()
    runtime.run(hours=kwargs.pop("hours", 120.0), record_interval=2.0, verbose=kwargs.pop("verbose", True), orbital_export=False)
    runtime.save_outputs(prefix=kwargs.pop("prefix", "final"), export_orbital=True)
    return runtime.build_orbital_bundle(delta_t=max(runtime.t_h, runtime.dt_h), prefix="final")


def default_habitat_protocol(outdir: str = "outputs_abiogenesis_habitat", **kwargs):
    return run_habitat_scan(outdir=outdir, Nx=kwargs.pop("Nx", 96), Ny=kwargs.pop("Ny", 96), dt_h=kwargs.pop("dt_h", 0.05), **kwargs)


def default_feasibility_protocol(outdir: str = "outputs_abiogenesis_feasibility", **kwargs):
    return run_feasibility_scan(
        outdir=outdir,
        Nx=kwargs.pop("Nx", 64),
        Ny=kwargs.pop("Ny", 64),
        dt_h=kwargs.pop("dt_h", 0.05),
        hours=kwargs.pop("hours", 60.0),
        enable_multiproc=kwargs.pop("enable_multiproc", False),
        workers=kwargs.pop("workers", 4),
        **kwargs,
    )


def default_origin_comparison_protocol(outdir: str = "outputs_abiogenesis_comparison", **kwargs):
    return run_origin_comparison(
        outdir=outdir,
        Nx=kwargs.pop("Nx", 96),
        Ny=kwargs.pop("Ny", 96),
        dt_h=kwargs.pop("dt_h", 0.05),
        base_hours=kwargs.pop("base_hours", 120.0),
        **kwargs,
    )


__all__ = [
    "default_single_origin_protocol",
    "default_habitat_protocol",
    "default_feasibility_protocol",
    "default_origin_comparison_protocol",
]
