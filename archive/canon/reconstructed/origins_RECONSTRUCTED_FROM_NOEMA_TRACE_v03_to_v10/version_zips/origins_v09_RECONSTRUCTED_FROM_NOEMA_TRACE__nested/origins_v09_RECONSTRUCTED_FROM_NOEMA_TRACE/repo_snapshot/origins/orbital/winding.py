from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable


@dataclass
class WindingComponents:
    winding_ec: float
    winding_zs: float
    winding_rel: float
    winding_red: float

    @property
    def winding_number(self) -> float:
        return self.winding_ec + self.winding_zs + self.winding_rel + self.winding_red

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data["winding_number"] = self.winding_number
        return data


def _component_winding(phase_deltas: Iterable[float], time_ratios: Iterable[float]) -> float:
    total = 0.0
    for dphi, ratio in zip(phase_deltas, time_ratios):
        total += dphi * ratio
    return total / (2.0 * 3.141592653589793)


def compute_winding_components(dphi_ec: Iterable[float], dphi_zs: Iterable[float], dphi_rel: Iterable[float], dphi_red: Iterable[float], delta_t: float, tau_local_steps: Iterable[float]) -> WindingComponents:
    ratios = [delta_t / max(1e-9, tau) for tau in tau_local_steps]
    return WindingComponents(
        winding_ec=_component_winding(dphi_ec, ratios),
        winding_zs=_component_winding(dphi_zs, ratios),
        winding_rel=_component_winding(dphi_rel, ratios),
        winding_red=_component_winding(dphi_red, ratios),
    )
