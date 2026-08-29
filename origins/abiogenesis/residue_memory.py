"""Canonical abiogenesis residue-memory aliases over the orbital memory substrate."""

from ..orbital.memory import MemoryState, ReductionResidue, apply_memory_update

HistoricalResidue = ReductionResidue
HistoricalMemory = MemoryState
apply_residue_update = apply_memory_update

__all__ = ["HistoricalResidue", "HistoricalMemory", "apply_residue_update"]
