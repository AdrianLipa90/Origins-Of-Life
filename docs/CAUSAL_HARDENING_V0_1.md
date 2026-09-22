# Origins Causal Hardening v0.1

Status: `CANDIDATE / APPEND-ONLY / NO DNA INTEGRATION`

Base repository: `AdrianLipa90/Origins-Of-Life`  
Base main: `a23bb8c73e3370566f514fecb9fa92b00c5a05ca`  
Working branch: `fix/gremlin-causal-hardening-v01`

## Scope

This hardening pass repairs internal causal and numerical consistency in
Origins-Of-Life. It deliberately does not integrate Origins-DNA; sequence-level
work remains an independent program.

## Epistemic layers

The active code now distinguishes three different claim types:

1. **Phenomenological chemistry/model dynamics** — explicit state transitions,
   reaction-rate approximations, diffusion, polymer and membrane fields.
2. **Geometric candidate layer** — synthetic topology, CP1/Bloch mapping and
   discrete geometric-phase readout. Physical binding remains open.
3. **Historical NOEMA lineage** — archived v03-v10 theory/results retained as
   project history, not silently promoted into current runtime evidence.

## Causal invariants introduced

- oligomer cleavage conserves nucleotide-equivalent units;
- NaN/Inf fails loudly instead of being replaced by random noise;
- clay adsorption transfers material into an explicit surface reservoir;
- surface concentration changes kinetics, not material quantity;
- polymerization consumes the same substrate amount that becomes polymer;
- population replication/fragmentation no longer creates R-field material;
- protocell count is a connected-component count, with occupied area tracked
  separately;
- post-operator protocell detection observes the same state that is recorded;
- orbital semantic mass does not ingest expected_protocells;
- orbital external load is a bounded coverage fraction rather than raw count;
- live topology coherence is bound before OORP/potential/winding derivation;
- repeated bundle construction retains relational memory;
- pure positive amplitude scaling must not create geometric phase through
  per-step curvature normalization;
- the zeta-indexed operator is documented and tested as an experimental spectral
  notch regularizer, with a grid-resolution-independent frequency mapping.

## Model boundary

The default UniversalOriginSimulator still starts in `PRESEEDED_RNA` mode for
backward compatibility. This is an explicit initial condition, not a simulation
of RNA arising from monomers.

A polymer-free start is now available with:

```python
UniversalOriginSimulator(..., preseed_rna=False)
```

The separate `origins.biology.first_rna` module is the monomer/oligomer
candidate path. It is hardened here but not yet silently spliced into the
canonical spatial simulator.

## Validation gates

The pull-request workflow compiles the active runtime and executes the complete
active `tests/` suite. A green semantic-facade smoke test alone is no longer
sufficient for this branch.

## Non-claims

This pass does not establish that:

- CP1/Bloch geometry is a physical mechanism of abiogenesis;
- zeta-zero indexing is a physical constraint on chemistry;
- a simulated protocell candidate is empirical abiogenesis;
- historical emergence times are preserved after conservation fixes.

Changed outputs must be treated as new model results, not retroactive corrections
to archived runs.
