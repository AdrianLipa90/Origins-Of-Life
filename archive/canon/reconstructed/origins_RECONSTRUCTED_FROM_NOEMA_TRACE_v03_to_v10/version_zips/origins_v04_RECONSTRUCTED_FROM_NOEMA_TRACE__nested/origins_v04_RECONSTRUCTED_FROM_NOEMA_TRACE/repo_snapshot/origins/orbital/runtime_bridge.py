from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

from ..bindings import scenario_config_to_entity_record
from ..scenarios import ScenarioConfig
from ..simulator.universal import UniversalOriginSimulator
from .memory import MemoryState
from .oorp import OORPTrace, run_oorp_pipeline
from .potentials import PotentialTerms, compute_potential_terms
from .repository_assignment import assign_orbital_state_to_entity
from .state import OrbitalCoordinate
from .winding import WindingComponents, compute_winding_components


@dataclass
class OrbitalRunBundle:
    entity_record: Dict[str, object]
    coordinate: Dict[str, object]
    potentials: Dict[str, object]
    winding: Dict[str, object]
    oorp_trace: Dict[str, object]
    memory_state: Dict[str, object]
    outputs: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "entity_record": dict(self.entity_record),
            "coordinate": dict(self.coordinate),
            "potentials": dict(self.potentials),
            "winding": dict(self.winding),
            "oorp_trace": dict(self.oorp_trace),
            "memory_state": dict(self.memory_state),
            "outputs": dict(self.outputs),
        }


class OrbitalRuntimeBridge:
    def __init__(self, simulator: UniversalOriginSimulator):
        self.simulator = simulator

    @classmethod
    def from_config(
        cls,
        config: ScenarioConfig,
        Nx: int = 96,
        Ny: int = 96,
        dt_h: float = 0.05,
        outdir: str = "outputs",
        include_clay: bool = True,
    ) -> "OrbitalRuntimeBridge":
        sim = UniversalOriginSimulator(
            config,
            Nx=Nx,
            Ny=Ny,
            dt_h=dt_h,
            outdir=outdir,
            include_clay=include_clay,
        )
        return cls(sim)

    def _build_coordinate(self, delta_t: float) -> OrbitalCoordinate:
        record = scenario_config_to_entity_record(self.simulator.config, source_path="origins/scenarios.py")
        return assign_orbital_state_to_entity(record, delta_t=delta_t)

    def _build_potentials(self, coordinate: OrbitalCoordinate) -> PotentialTerms:
        return compute_potential_terms(
            coherence=coordinate.coherence,
            defect=coordinate.defect,
            relation_depth=coordinate.relation_depth,
            semantic_mass=coordinate.semantic_mass,
            memory_affinity=0.0,
            external_load=float(self.simulator.protocell_count),
        )

    def _build_winding(self, coordinate: OrbitalCoordinate, reduction_score: float) -> WindingComponents:
        return compute_winding_components(
            dphi_ec=[coordinate.phi],
            dphi_zs=[coordinate.defect],
            dphi_rel=[coordinate.coherence],
            dphi_red=[reduction_score],
            delta_t=max(1e-9, coordinate.tau_local),
            tau_local_steps=[coordinate.tau_local],
        )

    def run(
        self,
        hours: float = 120.0,
        record_interval: float = 2.0,
        verbose: bool = True,
        save_outputs: bool = True,
        prefix: str = "final",
        export_orbital: bool = True,
    ) -> OrbitalRunBundle:
        self.simulator.initialize()
        self.simulator.run(hours=hours, record_interval=record_interval, verbose=verbose)
        if save_outputs:
            self.simulator.save_outputs(prefix=prefix)

        entity_record = scenario_config_to_entity_record(self.simulator.config, source_path="origins/scenarios.py")
        coordinate = self._build_coordinate(delta_t=hours)
        memory_state = MemoryState()
        oorp_trace = run_oorp_pipeline(coordinate, memory_state, external_load=float(self.simulator.protocell_count))
        potentials = self._build_potentials(coordinate)
        winding = self._build_winding(coordinate, oorp_trace.reduction_score)

        coordinate.omega = winding.winding_number

        outputs = {
            "history_csv": os.path.join(self.simulator.outdir, f"{prefix}_history.csv"),
            "fields_npz": os.path.join(self.simulator.outdir, f"{prefix}_fields.npz"),
            "heatmap_png": os.path.join(self.simulator.outdir, f"{prefix}_heatmaps.png"),
            "summary_csv": os.path.join(self.simulator.outdir, "summary.csv"),
        }

        bundle = OrbitalRunBundle(
            entity_record=entity_record.to_dict(),
            coordinate=coordinate.to_dict(),
            potentials=potentials.to_dict(),
            winding=winding.to_dict(),
            oorp_trace=oorp_trace.to_dict(),
            memory_state=memory_state.to_dict(),
            outputs=outputs,
        )

        if export_orbital:
            self.export_bundle(bundle, prefix=prefix)

        return bundle

    def export_bundle(self, bundle: OrbitalRunBundle, prefix: str = "final") -> str:
        path = os.path.join(self.simulator.outdir, f"{prefix}_orbital_bundle.json")
        os.makedirs(self.simulator.outdir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle.to_dict(), f, indent=2, ensure_ascii=False)
        return path
