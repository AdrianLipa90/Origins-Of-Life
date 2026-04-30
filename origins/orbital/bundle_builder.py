from __future__ import annotations

import json
import os

from ..bindings import scenario_config_to_entity_record
from .memory import MemoryState
from .oorp import run_oorp_pipeline
from .potentials import compute_potential_terms
from .repository_assignment import assign_orbital_state_to_entity
from .runtime_bridge import OrbitalRunBundle
from .winding import compute_winding_components


def build_orbital_bundle_from_simulator(simulator, delta_t: float, prefix: str = "final") -> OrbitalRunBundle:
    entity_record = scenario_config_to_entity_record(simulator.config, source_path="origins/scenarios.py")
    coordinate = assign_orbital_state_to_entity(entity_record, delta_t=delta_t)
    memory_state = MemoryState()
    oorp_trace = run_oorp_pipeline(coordinate, memory_state, external_load=float(simulator.protocell_count))
    potentials = compute_potential_terms(
        coherence=coordinate.coherence,
        defect=coordinate.defect,
        relation_depth=coordinate.relation_depth,
        semantic_mass=coordinate.semantic_mass,
        memory_affinity=0.0,
        external_load=float(simulator.protocell_count),
    )
    winding = compute_winding_components(
        dphi_ec=[coordinate.phi],
        dphi_zs=[coordinate.defect],
        dphi_rel=[coordinate.coherence],
        dphi_red=[oorp_trace.reduction_score],
        delta_t=max(1e-9, coordinate.tau_local),
        tau_local_steps=[coordinate.tau_local],
    )
    # Berry holonomy from topology field (Bloch sphere accumulated phase)
    berry_topo = float(getattr(getattr(simulator, "topo", None), "berry_accumulated", 0.0))
    if hasattr(getattr(simulator, "topo", None), "bloch_coherence"):
        coordinate.coherence = simulator.topo.bloch_coherence()
        coordinate.defect = 1.0 - coordinate.coherence
    coordinate.omega = winding.winding_number + berry_topo

    outputs = {
        "history_csv": os.path.join(simulator.outdir, f"{prefix}_history.csv"),
        "fields_npz": os.path.join(simulator.outdir, f"{prefix}_fields.npz"),
        "heatmap_png": os.path.join(simulator.outdir, f"{prefix}_heatmaps.png"),
        "summary_csv": os.path.join(simulator.outdir, "summary.csv"),
    }
    return OrbitalRunBundle(
        entity_record=entity_record.to_dict(),
        coordinate=coordinate.to_dict(),
        potentials=potentials.to_dict(),
        winding=winding.to_dict(),
        oorp_trace=oorp_trace.to_dict(),
        memory_state=memory_state.to_dict(),
        outputs=outputs,
    )


def export_orbital_bundle_from_simulator(simulator, delta_t: float, prefix: str = "final") -> str:
    bundle = build_orbital_bundle_from_simulator(simulator, delta_t=delta_t, prefix=prefix)
    os.makedirs(simulator.outdir, exist_ok=True)
    path = os.path.join(simulator.outdir, f"{prefix}_orbital_bundle.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle.to_dict(), f, indent=2, ensure_ascii=False)
    return path
