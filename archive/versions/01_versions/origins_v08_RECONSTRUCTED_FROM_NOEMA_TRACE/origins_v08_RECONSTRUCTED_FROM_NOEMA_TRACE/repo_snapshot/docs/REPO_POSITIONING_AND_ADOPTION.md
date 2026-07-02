# Repository positioning and adoption checklist

## Current semantic tension

The repository currently contains three coexisting frames:

1. legacy high-level CIEL/0 + CMB framing in `README.md`
2. executable orbital substrate in `origins.orbital`
3. canonical repo-facing abiogenesis surface in `origins.abiogenesis`

This is workable only if their roles remain clearly separated.

## Canonical decision

For day-to-day use, teaching, examples and future docs:

- prefer `origins.abiogenesis`
- prefer `scripts/run_abiogenesis.py`
- refer to scenario comparison and habitat feasibility in repo-native terms

For low-level execution and internal mechanics:

- keep `origins.orbital` as substrate
- keep orbital names internal where already integrated

## Adoption order

### Phase 1 — public docs
- point users to `START_HERE_ORIGINS.md`
- point users to `docs/ABIOGENESIS_PUBLIC_SURFACE.md`
- avoid introducing new examples that start from `origins.orbital` (internal substrate)

### Phase 2 — examples and tutorials
- all new examples should import from `origins.abiogenesis` or `origins.abiogenesis.api`
- all new shell commands should prefer `scripts/run_abiogenesis.py`

### Phase 3 — package ergonomics
- if repository tooling later allows safe update of existing files, export the canonical surface from package-level `__init__` files
- until then, use the explicit canonical modules already added

## Rejection rule

Reject changes that:
- add a second repo-facing semantic surface equal to `abiogenesis`
- reintroduce 1:1 public aliases with no semantic gain
- promote substrate names as the default user-facing language

## Success criterion

A new contributor should be able to understand the repository as an abiogenesis / emergence simulator **without first learning the orbital implementation vocabulary**.