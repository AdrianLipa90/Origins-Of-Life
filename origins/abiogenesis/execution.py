"""Canonical abiogenesis execution aliases over the orbital runtime substrate."""

from ..orbital.runtime_bridge import OrbitalRunBundle, OrbitalRuntimeBridge

AbiogenesisRunBundle = OrbitalRunBundle
AbiogenesisRuntimeAdapter = OrbitalRuntimeBridge

__all__ = ["AbiogenesisRunBundle", "AbiogenesisRuntimeAdapter"]
