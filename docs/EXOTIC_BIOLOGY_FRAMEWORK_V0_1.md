# Exotic Biology Framework v0.1

Status: `CANDIDATE / SUBSTRATE-AGNOSTIC / NO EXOTIC RUNTIME CLAIM`

Repository: `AdrianLipa90/Origins-Of-Life`

## Purpose

The repository previously changed solvent, temperature, pressure and kinetic
parameters while continuing to use the same water/RNA/lipid runtime.  That made
scenarios C and D environmental variants of a terrestrial reference model, not
implemented exotic biologies.

v0.1 separates:

```text
WorldEnvironment != BiochemistryProfile
```

A physical world can constrain a biology hypothesis without defining it.

## Relational life invariants

The framework does not define life as "water + RNA + lipids".  Every candidate
biology must instead satisfy the same functional contract:

1. `BOUNDED_SYSTEM`
2. `ENERGY_THROUGHPUT`
3. `PERSISTENT_INFORMATION_STATE`
4. `HERITABLE_STATE_TRANSFORMATION`
5. `SELECTION_OR_DIFFERENTIAL_PERSISTENCE`

These are model invariants, not a claim that this list is a complete scientific
definition of life.

## Profiles

### WATER_REFERENCE

Epistemic status: `REFERENCE_MODEL`  
Runtime status: `REFERENCE_IMPLEMENTED`

This is the existing water/RNA-like/amphiphile computational reference.
"Reference" means implemented baseline, not empirical proof of abiogenesis.

### AMMONIA_CANDIDATE

Epistemic status: `CANDIDATE`  
Runtime status: `TERRACENTRIC_CONTROL_ONLY`

Open components include:
- information carrier;
- solvent-compatible compartment boundary;
- building-block chemistry;
- replication/inheritance operator;
- solvent-specific transport and degradation laws.

The current universal simulator may be run in the ammonia world only as a
terracentric control.  Its RNA/lipid outputs are not ammonia-life predictions.

### HYDROCARBON_CANDIDATE

Epistemic status: `CANDIDATE`  
Runtime status: `TERRACENTRIC_CONTROL_ONLY`

Open components include:
- non-polar-compatible information-bearing state;
- persistent hydrocarbon-compatible boundary;
- building-block chemistry;
- inheritance operator;
- solvent-specific transport and reaction laws.

The current universal simulator may be run in the methane/ethane world only as
a terracentric control.  Its RNA/lipid outputs are not Titan-life predictions.

## Scenario binding

- A: `WATER_REFERENCE`
- B: `WATER_REFERENCE`
- C: `AMMONIA_CANDIDATE`
- D: `HYDROCARBON_CANDIDATE`
- E: `WATER_REFERENCE`

The binding is explicit and independently validated against the world solvent.

## Reporting rule

For candidate exotic profiles:

- `Exotic_Biology_Simulated = false`
- `Biology_Runtime_Status = TERRACENTRIC_CONTROL_ONLY`
- `Expected_ProtoC = null`
- `Success_Rate_pct = null`

The repository therefore cannot silently compare a candidate exotic biology to
a terrestrial expected-protocell target.

## Next implementation layer

A future exotic runtime should implement profile-specific operators for:

```text
environment transport
    -> building blocks
    -> persistent information carrier
    -> heritable transformation
    -> compartment boundary
    -> selection / differential persistence
```

Only after those operators exist and pass conservation/no-go gates may a
candidate profile move away from `TERRACENTRIC_CONTROL_ONLY`.

## Non-claims

v0.1 does not establish that:
- ammonia-based life exists;
- methane/ethane-based life exists;
- any particular exotic information polymer is viable;
- any particular exotic membrane chemistry is viable;
- the five relational invariants are a complete definition of life.

The purpose of v0.1 is architectural and epistemic: make future exotic biology
falsifiable without smuggling terrestrial biology into the result.
