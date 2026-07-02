"""Repo-native recurrence facade over internal orbital winding."""
from ..orbital.winding import WindingComponents as RecurrenceComponents
from ..orbital.winding import compute_winding_components as compute_recurrence_components

__all__ = ["RecurrenceComponents", "compute_recurrence_components"]
