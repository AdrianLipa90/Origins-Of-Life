# origins_planetary_biology_app

Mini repo for **first-biology / abiogenesis simulations under planetary conditions**.

This is the corrected side-application form of the `Origin*.*` line:
- planetary scenarios first,
- prebiotic chemistry second,
- topology / phase modulation third,
- protocell / RNA emergence metrics as output.

## Core architecture

```text
planet / environment
-> scenario configuration
-> chemistry + catalysis
-> topology / phase / constraints
-> protocell and RNA emergence metrics
```

## Included scenario families

- A — shallow UV + clay
- B — hydrothermal vents
- C — ammonia worlds
- D — Titan methane lakes
- E — Enceladus subsurface ocean

## Repo layout

- `origins/` — modular core package
- `app/` — thin CLI / smoke-run layer
- `results/` — example outputs
- `legacy/` — lineage files from monolithic and extended versions
- `METHOD.md` — method and assumptions
- `registry.yaml` — local registry

## Quick start

```bash
pip install -e .
python -m app.cli --scenario A --hours 4 --nx 32 --ny 32
```

## What this repo is

A **planetary first-biology side-app**.

## What this repo is not

It is not a finished biological theory of life.
It is an executable scenario simulator for comparing abiogenesis conditions across worlds.
