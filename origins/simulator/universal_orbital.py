from __future__ import annotations

from .universal import UniversalOriginSimulator
from ..bindings import scenario_config_to_entity_record
from ..orbital.bundle_builder import (
    build_orbital_bundle_from_simulator,
    export_orbital_bundle_from_simulator,
)


class OrbitalUniversalOriginSimulator(UniversalOriginSimulator):
    """
    Native orbital extension of UniversalOriginSimulator.

    This class keeps the original simulation path intact while adding
    orbital-native methods for entity records, orbital bundles and
    post-run orbital export.
    """

    def build_entity_record(self):
        return scenario_config_to_entity_record(self.config, source_path="origins/scenarios.py")

    def build_orbital_bundle(self, delta_t: float | None = None, prefix: str = "final"):
        if delta_t is None:
            delta_t = max(self.t_h, self.dt_h)
        return build_orbital_bundle_from_simulator(self, delta_t=delta_t, prefix=prefix)

    def export_orbital_bundle(self, delta_t: float | None = None, prefix: str = "final") -> str:
        if delta_t is None:
            delta_t = max(self.t_h, self.dt_h)
        return export_orbital_bundle_from_simulator(self, delta_t=delta_t, prefix=prefix)

    def run(
        self,
        hours: float = 120.0,
        record_interval: float = 2.0,
        verbose: bool = True,
        orbital_export: bool = False,
        orbital_prefix: str = "final",
    ):
        df = super().run(hours=hours, record_interval=record_interval, verbose=verbose)
        if orbital_export:
            self.export_orbital_bundle(delta_t=max(hours, self.t_h), prefix=orbital_prefix)
        return df

    def save_outputs(self, prefix: str = "final", export_orbital: bool = False) -> None:
        super().save_outputs(prefix=prefix)
        if export_orbital:
            self.export_orbital_bundle(delta_t=max(self.t_h, self.dt_h), prefix=prefix)
