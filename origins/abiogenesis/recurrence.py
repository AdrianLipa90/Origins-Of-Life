"""Canonical abiogenesis recurrence aliases over the orbital winding substrate."""

from ..orbital.winding import WindingComponents, compute_winding_components

RecurrenceComponents = WindingComponents
compute_recurrence_components = compute_winding_components

__all__ = ["RecurrenceComponents", "compute_recurrence_components"]
