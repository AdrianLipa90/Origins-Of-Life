# First-RNA Reaction-Regime Representation Gate v0.3 — Results

Status: `PASS_REACTION_REGIME_REPRESENTATION / APPEND-ONLY`

Date: 2026-09-22  
Execution head: `0e1f2ff4911e21d0090d03ea5d5458b66c51509b`  
Artifact id: `10703094926`  
Artifact SHA256: `ec1309d1339fd410bb7e2c871fed2f3dbc42126d4b9fc326c24e74e2e139f4dc`

Validation:

- focused representation tests: `8 passed`;
- full active repository suite: `106 passed`;
- Caimi-2025 pH-11 AUGC signature -> `CAIMI_2025_AUGC_PH11`;
- Song-2024 UMP hot-acid signature -> `SONG_2024_UMP_HOT_ACID`;
- legacy First-RNA v0.2 signature -> `UNBOUND`;
- nucleotide-mass fractions are scale invariant;
- QT45 remains a separate downstream function anchor;
- no kinetic parameter was fitted;
- no topology, zeta, or replication term was used.

## Consequence

The earlier label "missing activation chemistry" is generalized to
`reaction microenvironment / reactive state`.

This is required by the empirical distinction between at least two demonstrated
wet-dry RNA-polymerization regimes:

1. mild-alkaline 2',3'-cyclic NMP chemistry with high conversion but a shorter
   detected chain-length tail;
2. hot-acid ordinary NMP thin-film chemistry with a substantially longer
   detected tail.

The legacy `65 C / pH 7.5 / generic clay scalar` state is not silently assigned
to either regime.

## Next gate

The next stage is the first conservative population-balance kinetic kernel.

Rules:

- calibrate only within one empirical regime;
- use nucleotide-mass observables, not arbitrary absolute model count;
- fit only a declared subset of observables;
- validate on withheld tail observables without retuning;
- preserve nucleotide-equivalent material;
- do not use topology, zeta, or replication to rescue chemistry.
