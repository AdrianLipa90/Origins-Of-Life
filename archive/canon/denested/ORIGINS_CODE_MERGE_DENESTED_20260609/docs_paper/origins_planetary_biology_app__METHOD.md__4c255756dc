# METHOD

## Purpose

Build a small, clean, pluggable repo for simulating **prebiotic emergence under planetary conditions**.

The design goal is not a universal ontology of life, but an executable comparative tool:

- choose a world,
- instantiate its environmental parameters,
- simulate chemistry, topology, and protocell/RNA emergence,
- compare outcomes.

## Method layers

### 1. Scenario layer
From `origins/scenarios.py`:
- environment
- temperature
- pressure
- UV flux
- solvent
- pH
- redox state
- catalyst
- kinetic coefficients
- topological pattern / strength / time dependence

### 2. Chemistry layer
From `origins/chemistry/`:
- diffuse fields
- solar / tide envelopes
- clay catalysis
- local concentration effects

### 3. Biology layer
From `origins/biology/`:
- RNA sequence / RNA population
- protocell detection

### 4. Topology layer
From `origins/topology/`:
- Kähler-Berry-Euler-like field
- curvature modulation
- Zeta-Riemann soft constraints
- Heisenberg-like noise injection

### 5. Simulator layer
From `origins/simulator/universal.py`:
- integrates all layers into one scenario run
- records history and emergence metrics

## Why this is the correct form

This side-app is centered on:
- worlds,
- conditions,
- emergence,
- comparison.

It is therefore the correct descendant of `Origin*.*` for planetary first-biology research.

## Legacy handling

The older monolithic and extended files are not thrown away.
They are preserved in `legacy/` as:
- historical line of development,
- source of future grafts,
- fallback reference if modular structure misses a behavior.
