# Ammonia Magneto-Induction Candidate v0.1

Status: `COMPUTATIONAL_CANDIDATE / PHYSICAL_BINDING_OPEN`

## Motivation

For Jovian ocean worlds the relevant magnetic mechanism is not a static
magnetic field by itself. A time-varying external field can induce electric
fields and currents in a conductive subsurface liquid.

The candidate therefore follows the structural chain:

```text
dB/dt
  -> induced electric field
  -> conductive current
  -> ohmic energy throughput
```

The ammonia branch keeps conductivity explicit because magnetic coupling depends
on the electrical properties of the liquid phase, not on the label "ammonia"
alone.

## Jovian context

Galileo magnetometer observations established magnetic induction as one of the
strongest lines of evidence for conductive subsurface oceans at Europa and
Callisto. Ganymede is more complicated because it also has an intrinsic
magnetic field.

Ammonia is not established as the dominant solvent of those oceans. However,
H2O-NH3 mixtures have long been modeled for colder Galilean moons because NH3
lowers the melting point and can help preserve liquid reservoirs. Ganymede and
Callisto are therefore useful physical analogs for a future ammoniacal-ocean
environment profile, not evidence that pure-NH3 life exists there.

## Implemented candidate

Mode:

`JOVIAN_TIME_VARYING_FIELD_CANDIDATE`

Baseline:

`MAGNETO_INDUCTION_OFF`

The default remains OFF so all previous ammonia results remain reproducible.

The normalized driver computes:

```text
E_ind ~ (L/2) dB/dt
J     ~ sigma E_ind
P_ohm ~ sigma E_ind^2
```

and couples a bounded fraction of the resulting power proxy into the open
energy field E.

A non-zero but static magnetic field produces zero induction source. Zero
conductivity also produces zero current and zero ohmic source.

## Parameter status

All field amplitudes, frequencies, conductivity, length scale and coupling gain
are:

`UNVALIDATED_DIMENSIONLESS_CANDIDATE`

They are not calibrated Jovian field strengths, ocean conductivities or orbital
frequencies.

## Scope

This implementation changes only the open energy sector. It does not create or
destroy material in the conserved P + I + B sector.

It does not yet implement:
- Lorentz-force-driven fluid transport;
- spatial current loops;
- magnetohydrodynamic convection;
- measured H2O-NH3 conductivity;
- a specific Ganymede or Callisto interior model.

Those require a separately calibrated environment layer.

## References

- NASA Europa Clipper, "Induced Magnetic Field from Europa's Subsurface Ocean".
- Zimmer, Khurana & Kivelson, Icarus 147 (2000), 329-347,
  doi:10.1006/icar.2000.6456.
- "Electrical Properties of Icy World Oceans from Laboratory Measurements",
  ACS Earth and Space Chemistry 10 (2026), 1189-1200,
  doi:10.1021/acsearthspacechem.5c00333.
- "Ammonia as a parameter shaping habitability on icy moons",
  FEMS Microbes (2026), doi:10.1093/femsmc/xtag015.

## Non-claims

This module does not establish:
- that ammonia-based life exists;
- that a Jovian moon contains a pure-ammonia biosphere;
- that magnetic induction is sufficient for abiogenesis;
- that the normalized driver has quantitative predictive power before
  calibration against measured field and conductivity data.
