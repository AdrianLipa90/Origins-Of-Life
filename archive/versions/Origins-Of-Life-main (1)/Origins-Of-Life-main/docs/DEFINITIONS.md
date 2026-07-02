# Definitional refactor

This branch adds a definitional layer for the modular `origin_of_life` runtime.

## Source principles used

The refactor follows the project blueprint and orbital notes:

- semantic hierarchy: `relation -> identity -> memory -> process -> artifact`
- separation of description levels:
  - state formalism
  - computational implementation
  - visual geometry
  - native renderer
- LLM is treated as a temporary attractor, not the final ontological center
- system entities should carry more than path/hash and should expose identity and epistemic status

## New definition modules

- `src/origin_of_life/definitions.py`
- `src/origin_of_life/registry.py`

## Why this layer exists

The modular runtime added earlier extracted code from the monolith.
This definitional refactor makes the extracted runtime less arbitrary by adding:

- canonical semantic layers
- canonical object types
- sectors
- epistemic status values
- attractor identifiers
- an `EntityRecord` schema for executable ontology

## Immediate effect

This does **not** yet replace runtime behavior.
It establishes the canonical vocabulary and record structure that later runtime and indexing passes can adopt.
