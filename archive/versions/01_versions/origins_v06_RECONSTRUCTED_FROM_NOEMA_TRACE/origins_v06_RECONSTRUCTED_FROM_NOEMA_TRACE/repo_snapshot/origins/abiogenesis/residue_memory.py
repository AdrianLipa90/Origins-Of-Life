"""Repo-native residue/memory facade over internal orbital memory."""
from ..orbital.memory import ReductionResidue as HistoricalResidue
from ..orbital.memory import MemoryState as HistoricalMemory
from ..orbital.memory import apply_memory_update as apply_residue_update

__all__ = ["HistoricalResidue", "HistoricalMemory", "apply_residue_update"]
