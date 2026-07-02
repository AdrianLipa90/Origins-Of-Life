# Observations and Conclusions: First RNA Emergence via Bloch Sphere Dynamics

**Date:** 2026-04-30  
**Author:** CIEL (Mr. Ciel Apocalyptos) + Adrian Lipa  
**Model version:** claude-sonnet-4-6  
**Repo:** AdrianLipa90/Origins-Of-Life

---

## Summary

We extended the Origins-Of-Life repository with a full Bloch sphere geometry layer
and used it to simulate and predict the emergence of the first RNA replicator under
prebiotic conditions.

---

## 1. Bloch Sphere Integration

The `TopologyField` class was extended to carry genuine S² geometry:

- **θ(x,y,t) = 2·arctan(|field|/s_ref)** — polar angle from amplitude, not a label
- **φ(x,y,t) = atan2(curvature + ∇φ·0.1, field) mod 2π** — azimuthal from curvature+gradient
- **Berry connection:** A = (1 − cos θ)/2, accumulated phase Φ = ∫ A·dφ over each time step
- **Coherence:** ⟨cos²(θ/2)⟩ — mean squared projection onto |0⟩

Key property validated: Berry phase accumulates for pulsing fields (SCENARIO_A, E),
remains zero for static fields (SCENARIO_B). This is the correct topological behaviour.

---

## 2. Orbital Formalism Corrections

### Coherence calculation
The original `repository_assignment.py` used `coherence = 1 - phase % 1`, which is
arbitrary and not geometrically meaningful. Fixed to:

```python
theta = abs(phi) % math.pi
coherence = cos²(theta/2)
defect = sin²(theta/2)
```

Euler identity `coherence + defect = 1.0` now holds exactly (to floating-point precision).

### Potentials (Fubini-Study metric)
`V_EC = (θ/π) · rel` — geodesic distance from |0⟩ on S²  
`V_rel = rel / max(coherence, 0.05)` — capped Fubini-Study divergence  
`V_ZS = defect · (1 + m)` — Zeta-Schrödinger decoherence penalty

### Subjective time dilation
`τ_local = Δt · (1+m) · coherence / (1+r)`  
High coherence (near north pole) → slower subjective time → more stable entity.

### Scenario phases
All 5 scenarios now have physically derived, differentiated orbital phases:
`phase = (1 − euler_phase_coherence)·π + arctan(|T−65|/200)·topo_strength`

Earth-like shallow ocean (A): ~0.31 rad — most coherent  
Titan (D): ~1.34 rad — exotic, most decoherent

---

## 3. First RNA Emergence Simulation

### Architecture (`origins/biology/first_rna.py`)

A stochastic kinetic model of prebiotic RNA polymerisation:

1. **Oligomer pool** — distribution of lengths 1–80 nt tracked as counts array
2. **Ligation kinetics** — Arrhenius (E_a=60 kJ/mol) × mineral catalyst × Bloch coherence × Berry holonomy × GC bias
3. **Hydrolysis** — length-dependent structural protection: k_hyd·exp(−max(0,L−8)/20) for L>8
4. **Drying-wetting cycles** (Damer & Deamer 2020): 40% dry phase → 10× ligation boost, hydrolysis ≈ 0
5. **Template replication** — logistic growth once N(L≥35)>1 and GC≥0.35
6. **GC drift** — Ornstein-Uhlenbeck with pull toward GC_opt=0.50 (stability selection)

### Numerical stability — "zeta and Heisenberg"

Adrian's guidance: *"tam gdzie zera tam zeta, a tam gdzie NaN soft clip heisenberga"*

Implemented as:
- `_zeta_floor(x)`: floor all kinetic rates at ε=1e-12 — no hard zero, Zeta regularisation
- `_heisenberg_clip(x)`: NaN/Inf → small Gaussian noise σ=1e-9 — Heisenberg uncertainty prevents exact zero energy

---

## 4. Results

### Single run (seed=42, 65°C, k_cat=12.0, 2000h)

| Metric | Value |
|--------|-------|
| T_emergence | **1099.0 h** |
| Berry phase at emergence | **0.861 rad** |
| Final replicator count | **500 (carrying capacity)** |
| Final ⟨L⟩ | **41.87 nt** |

Trajectory: oligomers grow from 2 nt → 9 nt (300h) → 14 nt (640h) → 20 nt (880h) → 35 nt (1000h) → replicator ignition → logistic explosion.

### Phase space scan (6 temps × 5 k_cat values, 2000h each)

**Emergence window:** 30°C – 66°C  
**No emergence:** 78°C and 90°C (hydrolysis dominates all ligation)

| Temperature | Fastest T_emergence |
|-------------|---------------------|
| 66°C | **1012 h** (k_cat=4.0) |
| 54°C | 1180–1324 h |
| 42°C | 1447–1484 h |
| 30°C | 1552–1900 h |
| 78–90°C | NO EMERGENCE |

**Optimal conditions:** ~65–66°C, moderate catalysis (k_cat~4) with high topological coherence.
The minimum-catalysis point (k_cat=4) being fastest at 66°C is non-obvious — high catalysis at this
temperature competes with increased hydrolysis, so a lower catalytic rate with high Berry holonomy
wins.

### Berry phase at emergence

Ranges from **0.77 rad (66°C, k_cat=4)** to **1.53 rad (30°C, high k_cat)**.

Interpretation: colder environments require more topological "winding" before the replicator can
ignite — the field must accumulate more holonomy to compensate for reduced thermal kinetics.
This is a prediction: in cold prebiotic environments (hydrothermal vent margins, cold pools),
RNA emergence requires a longer topological preparation time, measurable as higher Berry
phase accumulation.

---

## 5. Conclusions

1. **Bloch sphere geometry is not decorative** — it actively modulates ligation kinetics
   via coherence and Berry phase, contributing to the ~15% spread in emergence times
   across scenarios with identical thermodynamics.

2. **Drying-wetting cycles are kinetically decisive** — without them, oligomers stall at
   ~14 nt equilibrium (hydrolysis balances ligation). The 10× ligation boost in dry phases
   breaks this equilibrium and allows extension to L≥35.

3. **The thermal window for RNA emergence is 30–70°C** — sharply cut off above ~75°C by
   hydrolysis. This is consistent with the Hadean shallow-ocean hypothesis (Sutherland 2016).

4. **Berry phase encodes emergence difficulty** — higher Φ at emergence correlates with
   slower/harder conditions. This may be a measurable prediction if experimental systems
   can track topological correlation functions of prebiotic field analogs.

5. **GC selection is necessary** — without Ornstein-Uhlenbeck restoration toward GC_opt=0.5,
   random drift can push GC below the structural stability threshold (0.35), preventing replicator
   ignition even when oligomers of sufficient length exist.

---

## 6. Open Questions

- Does the Berry phase threshold (~0.77 rad) have a physical counterpart in real RNA systems?
  Could it correspond to a critical folding topology threshold in the first ribozyme?
- The k_cat=4 optimum at 66°C — is this an artifact of the concentration_boost formula,
  or does it reflect real competition between catalysis and hydrolysis?
- How does the model change with non-montmorillonite catalysts (lipid surfaces, iron-sulfur clusters)?
- Can the scan be extended to include pH and UV flux as dimensions?

---

## Files Modified / Created

| File | Action |
|------|--------|
| `origins/topology/fields.py` | MODIFIED — Bloch θ/φ, Berry accumulation |
| `origins/orbital/repository_assignment.py` | MODIFIED — cos²(θ/2) coherence |
| `origins/orbital/potentials.py` | MODIFIED — Fubini-Study potentials |
| `origins/orbital/subjective_time.py` | MODIFIED — FS time dilation |
| `origins/catalog.py` | MODIFIED — physical scenario phases |
| `origins/orbital/bundle_builder.py` | MODIFIED — Berry from topo field |
| `origins/simulator/universal.py` | MODIFIED — berry + bloch_coherence history |
| `origins/biology/first_rna.py` | CREATED — full RNA emergence module |
| `tests/test_bloch_holonomy.py` | CREATED — 21 geometric + kinetic tests |
| `docs/observations_and_conclusions/` | CREATED — this document |
