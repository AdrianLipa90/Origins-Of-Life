from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .boundary_morphogenesis import EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE
from .compartments import _periodic_components, periodic_component_topology
from .factory import create_exotic_candidate_simulator
from .hydrocarbon_merger_neck import diagnose_merger_neck
from .information_morphogenesis import PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE


SCHEMA = "ORIGINS_HYDROCARBON_BRIDGE_FLUX_V0_1"

EVENT_BRIDGE_FLUX_RESOLVED = "BRIDGE_FLUX_RESOLVED"
EVENT_NO_MERGER_WITHIN_HORIZON = "NO_MERGER_WITHIN_HORIZON"
EVENT_MERGER_NOT_SINGLE_NEW_PIXEL = "MERGER_NOT_SINGLE_NEW_PIXEL"

DOMINANT_REACTION = "REACTION_NET_DOMINANT"
DOMINANT_DIFFUSION = "DIFFUSION_DOMINANT"
DOMINANT_MIXED = "MIXED_OR_NONPOSITIVE"


@dataclass(frozen=True)
class HydrocarbonBridgeFluxResult:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    last_closed_step: int | None
    first_loss_step: int | None
    event: str
    bridge_x: int | None
    bridge_y: int | None
    information_threshold: float
    information_start: float | None
    information_after_reaction: float | None
    information_final: float | None
    threshold_gap_before_step: float | None
    assembly_growth: float | None
    decay_loss: float | None
    reaction_net_delta: float | None
    diffusion_delta: float | None
    total_delta: float | None
    closure_error: float | None
    reaction_fraction_of_positive_delta: float | None
    diffusion_fraction_of_positive_delta: float | None
    dominant_positive_contribution: str | None
    source_rate_before_information_step: float | None
    aggregate_reservoir_before_information_step: float | None
    energy_before_information_step: float | None
    trait_before_information_step: float | None
    minimum_new_active_bridge_pixels: int | None
    physical_binding: str
    model_mutation_allowed: bool = False
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def audited_hydrocarbon_step(simulator) -> dict[str, np.ndarray | float]:
    """Advance exactly one hydrocarbon runtime step while exposing I flux terms.

    This follows HydrocarbonCandidateSimulator.step() in the same order and
    returns an audit decomposition. It does not change parameters, thresholds,
    or source budgets.
    """
    simulator._require_initialized()
    p = simulator.parameters
    material_before = simulator.material_total()

    information_start = np.asarray(simulator.I, dtype=float).copy()

    simulator.step_energy_throughput()
    simulator.step_interface_exchange()

    source = np.asarray(
        simulator.candidate_assembly_sources()["information"],
        dtype=float,
    ).copy()
    aggregate_before = np.asarray(simulator.A, dtype=float).copy()
    energy_before = np.asarray(simulator.E, dtype=float).copy()
    trait_before = np.asarray(simulator.Q, dtype=float).copy()

    request = source * p.dt
    assembly_growth = np.minimum(request, aggregate_before)

    simulator.step_information_and_selection()
    information_after_reaction = np.asarray(simulator.I, dtype=float).copy()

    decay_loss = (
        information_start
        + assembly_growth
        - information_after_reaction
    )
    if float(np.min(decay_loss)) < -1e-10:
        raise RuntimeError("audited decay reconstruction became negative")
    decay_loss = np.maximum(decay_loss, 0.0)

    simulator.step_boundary()
    information_pre_transport = np.asarray(simulator.I, dtype=float).copy()
    simulator.step_transport()
    information_final = np.asarray(simulator.I, dtype=float).copy()
    diffusion_delta = information_final - information_pre_transport

    simulator._validate_state()
    material_after = simulator.material_total()
    tolerance = 2e-10 * max(1.0, abs(material_before))
    if abs(material_after - material_before) > tolerance:
        raise FloatingPointError(
            "audited hydrocarbon step violated material conservation"
        )
    simulator.time += p.dt

    reaction_net = information_after_reaction - information_start
    total_delta = information_final - information_start
    closure_error = total_delta - (reaction_net + diffusion_delta)

    return {
        "information_start": information_start,
        "information_source_rate": source,
        "aggregate_before_information": aggregate_before,
        "energy_before_information": energy_before,
        "trait_before_information": trait_before,
        "assembly_growth": assembly_growth,
        "decay_loss": decay_loss,
        "information_after_reaction": information_after_reaction,
        "reaction_net_delta": reaction_net,
        "diffusion_delta": diffusion_delta,
        "information_final": information_final,
        "total_delta": total_delta,
        "closure_error": closure_error,
    }


def _component_records(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    labels, components = _periodic_components(mask)
    topology = periodic_component_topology(mask)
    topo_by_label = {
        int(record["label_id"]): record
        for record in topology["components"]
    }
    return labels, [
        {
            "label_id": int(label_id),
            "cells": frozenset(cells),
            "area_pixels": int(len(cells)),
            "noncontractible": bool(
                topo_by_label[label_id]["noncontractible"]
            ),
        }
        for label_id, cells in enumerate(components, start=1)
    ]


def _dominant_positive(reaction: float, diffusion: float) -> tuple[str, float, float]:
    positive_reaction = max(0.0, float(reaction))
    positive_diffusion = max(0.0, float(diffusion))
    total = positive_reaction + positive_diffusion
    if total <= 1e-15:
        return DOMINANT_MIXED, 0.0, 0.0
    rf = positive_reaction / total
    df = positive_diffusion / total
    if rf > 0.6:
        dominant = DOMINANT_REACTION
    elif df > 0.6:
        dominant = DOMINANT_DIFFUSION
    else:
        dominant = DOMINANT_MIXED
    return dominant, float(rf), float(df)


def run_hydrocarbon_bridge_flux_case(
    scenario,
    *,
    seed: int,
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> HydrocarbonBridgeFluxResult:
    if getattr(scenario, "biochemistry_profile", None) != "HYDROCARBON_CANDIDATE":
        raise ValueError("bridge-flux diagnostic requires HYDROCARBON_CANDIDATE")
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be positive")

    sim = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=int(seed),
        information_morphogenesis=PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        boundary_morphogenesis=EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
    )
    sim.initialize()

    last_closed_step: int | None = None
    previous_mask: np.ndarray | None = None
    previous_observation: dict[str, object] | None = None
    was_closed = False

    for step in range(1, int(horizon_steps) + 1):
        if was_closed:
            audit = audited_hydrocarbon_step(sim)
        else:
            audit = None
            sim.step()

        observation = sim.candidate_compartments()
        closed = int(observation["count"]) > 0
        current_mask = np.asarray(sim.I) >= float(
            sim.parameters.information_threshold
        )

        if closed:
            last_closed_step = step
            previous_mask = current_mask.copy()
            previous_observation = dict(observation)
        elif was_closed:
            if audit is None or previous_mask is None or previous_observation is None:
                raise RuntimeError("missing audited merger-step context")

            prev_labels, prev_records = _component_records(previous_mask)
            accepted = np.asarray(
                previous_observation["accepted_mask"],
                dtype=bool,
            )
            accepted_labels = sorted(
                int(value)
                for value in np.unique(prev_labels[accepted])
                if int(value) > 0
            )
            backgrounds = [
                record
                for record in prev_records
                if bool(record["noncontractible"])
            ]

            best: tuple[int, dict[str, object], dict[str, object], dict[str, object]] | None = None
            for label_id in accepted_labels:
                island = prev_records[label_id - 1]
                for background in backgrounds:
                    diagnosis = diagnose_merger_neck(
                        previous_mask,
                        current_mask,
                        island["cells"],
                        background["cells"],
                    )
                    cost = diagnosis["minimum_new_active_bridge_pixels"]
                    if (
                        diagnosis["current_descendant_noncontractible"]
                        and cost is not None
                    ):
                        candidate = (
                            int(cost),
                            island,
                            background,
                            diagnosis,
                        )
                        if best is None or candidate[0] < best[0]:
                            best = candidate

            claim = sim.claim_status()
            threshold = float(sim.parameters.information_threshold)

            if best is None:
                return HydrocarbonBridgeFluxResult(
                    schema=SCHEMA,
                    profile=str(claim["profile"]),
                    runtime=str(claim["runtime"]),
                    seed=int(seed),
                    horizon_steps=int(horizon_steps),
                    last_closed_step=last_closed_step,
                    first_loss_step=step,
                    event=EVENT_MERGER_NOT_SINGLE_NEW_PIXEL,
                    bridge_x=None,
                    bridge_y=None,
                    information_threshold=threshold,
                    information_start=None,
                    information_after_reaction=None,
                    information_final=None,
                    threshold_gap_before_step=None,
                    assembly_growth=None,
                    decay_loss=None,
                    reaction_net_delta=None,
                    diffusion_delta=None,
                    total_delta=None,
                    closure_error=None,
                    reaction_fraction_of_positive_delta=None,
                    diffusion_fraction_of_positive_delta=None,
                    dominant_positive_contribution=None,
                    source_rate_before_information_step=None,
                    aggregate_reservoir_before_information_step=None,
                    energy_before_information_step=None,
                    trait_before_information_step=None,
                    minimum_new_active_bridge_pixels=None,
                    physical_binding=str(claim["physical_binding"]),
                )

            _, island, background, diagnosis = best
            previous = np.asarray(previous_mask, dtype=bool)
            new_active = current_mask & ~previous

            current_labels, current_components = _periodic_components(current_mask)
            descendant_labels = {
                int(current_labels[x, y])
                for x, y in island["cells"]
                if int(current_labels[x, y]) > 0
            }
            descendant_cells: set[tuple[int, int]] = set()
            for label_id in descendant_labels:
                descendant_cells.update(current_components[label_id - 1])

            bridge_cells = sorted(
                (x, y)
                for x, y in descendant_cells
                if bool(new_active[x, y])
            )

            if (
                int(diagnosis["minimum_new_active_bridge_pixels"]) != 1
                or len(bridge_cells) != 1
            ):
                return HydrocarbonBridgeFluxResult(
                    schema=SCHEMA,
                    profile=str(claim["profile"]),
                    runtime=str(claim["runtime"]),
                    seed=int(seed),
                    horizon_steps=int(horizon_steps),
                    last_closed_step=last_closed_step,
                    first_loss_step=step,
                    event=EVENT_MERGER_NOT_SINGLE_NEW_PIXEL,
                    bridge_x=None,
                    bridge_y=None,
                    information_threshold=threshold,
                    information_start=None,
                    information_after_reaction=None,
                    information_final=None,
                    threshold_gap_before_step=None,
                    assembly_growth=None,
                    decay_loss=None,
                    reaction_net_delta=None,
                    diffusion_delta=None,
                    total_delta=None,
                    closure_error=None,
                    reaction_fraction_of_positive_delta=None,
                    diffusion_fraction_of_positive_delta=None,
                    dominant_positive_contribution=None,
                    source_rate_before_information_step=None,
                    aggregate_reservoir_before_information_step=None,
                    energy_before_information_step=None,
                    trait_before_information_step=None,
                    minimum_new_active_bridge_pixels=int(
                        diagnosis["minimum_new_active_bridge_pixels"]
                    ),
                    physical_binding=str(claim["physical_binding"]),
                )

            x, y = bridge_cells[0]
            start = float(audit["information_start"][x, y])
            after_reaction = float(
                audit["information_after_reaction"][x, y]
            )
            final = float(audit["information_final"][x, y])
            assembly = float(audit["assembly_growth"][x, y])
            decay = float(audit["decay_loss"][x, y])
            reaction = float(audit["reaction_net_delta"][x, y])
            diffusion = float(audit["diffusion_delta"][x, y])
            total = float(audit["total_delta"][x, y])
            error = float(audit["closure_error"][x, y])
            dominant, reaction_fraction, diffusion_fraction = (
                _dominant_positive(reaction, diffusion)
            )

            return HydrocarbonBridgeFluxResult(
                schema=SCHEMA,
                profile=str(claim["profile"]),
                runtime=str(claim["runtime"]),
                seed=int(seed),
                horizon_steps=int(horizon_steps),
                last_closed_step=last_closed_step,
                first_loss_step=step,
                event=EVENT_BRIDGE_FLUX_RESOLVED,
                bridge_x=int(x),
                bridge_y=int(y),
                information_threshold=threshold,
                information_start=start,
                information_after_reaction=after_reaction,
                information_final=final,
                threshold_gap_before_step=float(threshold - start),
                assembly_growth=assembly,
                decay_loss=decay,
                reaction_net_delta=reaction,
                diffusion_delta=diffusion,
                total_delta=total,
                closure_error=error,
                reaction_fraction_of_positive_delta=reaction_fraction,
                diffusion_fraction_of_positive_delta=diffusion_fraction,
                dominant_positive_contribution=dominant,
                source_rate_before_information_step=float(
                    audit["information_source_rate"][x, y]
                ),
                aggregate_reservoir_before_information_step=float(
                    audit["aggregate_before_information"][x, y]
                ),
                energy_before_information_step=float(
                    audit["energy_before_information"][x, y]
                ),
                trait_before_information_step=float(
                    audit["trait_before_information"][x, y]
                ),
                minimum_new_active_bridge_pixels=1,
                physical_binding=str(claim["physical_binding"]),
            )

        was_closed = closed

    claim = sim.claim_status()
    return HydrocarbonBridgeFluxResult(
        schema=SCHEMA,
        profile=str(claim["profile"]),
        runtime=str(claim["runtime"]),
        seed=int(seed),
        horizon_steps=int(horizon_steps),
        last_closed_step=last_closed_step,
        first_loss_step=None,
        event=EVENT_NO_MERGER_WITHIN_HORIZON,
        bridge_x=None,
        bridge_y=None,
        information_threshold=float(sim.parameters.information_threshold),
        information_start=None,
        information_after_reaction=None,
        information_final=None,
        threshold_gap_before_step=None,
        assembly_growth=None,
        decay_loss=None,
        reaction_net_delta=None,
        diffusion_delta=None,
        total_delta=None,
        closure_error=None,
        reaction_fraction_of_positive_delta=None,
        diffusion_fraction_of_positive_delta=None,
        dominant_positive_contribution=None,
        source_rate_before_information_step=None,
        aggregate_reservoir_before_information_step=None,
        energy_before_information_step=None,
        trait_before_information_step=None,
        minimum_new_active_bridge_pixels=None,
        physical_binding=str(claim["physical_binding"]),
    )


def run_matched_hydrocarbon_bridge_flux(
    scenario,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> list[HydrocarbonBridgeFluxResult]:
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    return [
        run_hydrocarbon_bridge_flux_case(
            scenario,
            seed=seed,
            horizon_steps=horizon_steps,
            Nx=Nx,
            Ny=Ny,
        )
        for seed in seed_list
    ]


def diagnostic_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "profile": "HYDROCARBON_CANDIDATE",
        "information_mode": PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        "boundary_mode": EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
        "decomposition": "DELTA_I = ASSEMBLY - DECAY + DIFFUSION",
        "diagnostic_only": True,
        "model_mutation_allowed": False,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
    }
