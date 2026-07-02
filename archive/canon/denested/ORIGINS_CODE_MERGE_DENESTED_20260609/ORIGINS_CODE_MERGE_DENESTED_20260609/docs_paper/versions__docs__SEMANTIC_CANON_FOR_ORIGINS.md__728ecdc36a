# Semantic canon for `Origins-Of-Life`

This document defines the naming canon for the repository.

## Rule 0

If two names describe the same executable or conceptual layer, the repository must keep **one canonical name** and treat all others as non-canonical.

No duplicate semantic layers should be merged.

---

## 1. Repository-native semantic frame

This repository is about **origins of life / abiogenesis / prebiotic emergence**.

Therefore the semantic canon must prefer names grounded in:

- origin
- emergence
- habitat
- feasibility
- residue
- historical memory
- recurrence
- transition

and not generic imported vocabulary when a repository-native term exists.

---

## 2. Canonical vocabulary for this repo

### Canonical semantic names

- `OriginHabitatShell`
- `EmergenceCoordinate`
- `EmergenceSystemState`
- `FeasibilityTerms`
- `EmergenceClock`
- `RecurrenceComponents`
- `HistoricalResidue`
- `HistoricalMemory`
- `EmergenceTrace`
- `AbiogenesisRunBundle`
- `AbiogenesisRuntimeAdapter`

These names are semantically native to the repo.

### Implementation layer names allowed underneath

The current implementation already contains orbital machinery. Those names remain valid as **implementation-internal** names:

- `OrbitalSphere`
- `OrbitalCoordinate`
- `OrbitalSystemState`
- `PotentialTerms`
- `WindingComponents`
- `MemoryState`
- `OORPTrace`

These are acceptable internally, but they should not multiply into parallel public APIs with the same meaning.

---

## 3. Canonical public surface

### Public semantic surface for repo-facing use

Prefer a **single public semantic surface**:

- `origins.abiogenesis`
- `origins.simulator.api`
- `origins.analysis.api`

### Internal implementation surface

Keep implementation details under:

- `origins.orbital`
- `origins.simulator.universal_orbital`
- `origins.analysis.sweep_orbital`

Interpretation:
- `origins.orbital` = implementation substrate
- `origins.abiogenesis` = repository-native semantic surface

---

## 4. Non-canonical duplicates that should not be introduced

The following pattern is forbidden:

- creating a new public name that is only a 1:1 alias of an existing public name
- exporting both names as equal-status public APIs
- introducing multiple repo-facing terms for the same exact concept

Examples of what should **not** happen:

- both `OrbitalUniversalOriginSimulator` and some second public class with identical behavior but different naming
- both `run_topo_sweep_orbital` and another public function that is only a renamed alias with no semantic difference
- both `MemoryState` and another public repo-facing type if they are structurally identical and equally exposed

---

## 5. Canon rule by layer

### Sphere / habitat layer
Canonical repo-facing name:
- `OriginHabitatShell`

Internal implementation may still use:
- `OrbitalSphere`

### State layer
Canonical repo-facing names:
- `EmergenceCoordinate`
- `EmergenceSystemState`

Internal implementation may still use:
- `OrbitalCoordinate`
- `OrbitalSystemState`

### Potential layer
Canonical repo-facing name:
- `FeasibilityTerms`

Internal implementation may still use:
- `PotentialTerms`

### Time layer
Canonical repo-facing name:
- `EmergenceClock`

### Winding / recurrence layer
Canonical repo-facing name:
- `RecurrenceComponents`

Internal implementation may still use:
- `WindingComponents`

### Memory layer
Canonical repo-facing names:
- `HistoricalResidue`
- `HistoricalMemory`

Internal implementation may still use:
- `ReductionResidue`
- `MemoryState`

### Transition layer
Canonical repo-facing name:
- `EmergenceTrace`

Internal implementation may still use:
- `OORPTrace`

### Runtime layer
Canonical repo-facing name:
- `AbiogenesisRuntimeAdapter`

Internal implementation may still use:
- orbital runtime bridge / orbital subclassing

---

## 6. Immediate repository policy

1. Do not merge branches that introduce semantic duplicates without a canon declaration.
2. When adapting terminology, first declare:
   - canonical public name
   - internal implementation name
   - whether aliasing is temporary or forbidden
3. Public docs and examples should use the repo-native canonical vocabulary.
4. Internal code may keep orbital terms where they are already deeply integrated, but those names should not proliferate into additional public layers.

---

## 7. Current decision

For `Origins-Of-Life`, the semantic canon is:

- **public repo-facing semantics**: `abiogenesis / emergence / habitat / residue / recurrence`
- **internal implementation substrate**: `orbital`

This means the next semantic adaptation step should be a **single canonical public facade**, not a parallel forest of aliases.
