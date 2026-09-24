from __future__ import annotations

from dataclasses import dataclass
import math


MAGNETO_INDUCTION_OFF = "MAGNETO_INDUCTION_OFF"
JOVIAN_TIME_VARYING_FIELD_CANDIDATE = "JOVIAN_TIME_VARYING_FIELD_CANDIDATE"

MAGNETO_INDUCTION_MODES = (
    MAGNETO_INDUCTION_OFF,
    JOVIAN_TIME_VARYING_FIELD_CANDIDATE,
)

PHYSICAL_BINDING = "OPEN"
PARAMETER_STATUS = "UNVALIDATED_DIMENSIONLESS_CANDIDATE"


@dataclass(frozen=True)
class MagnetoInductionParameters:
    """Normalized time-varying-field candidate inspired by Jovian induction.

    The structure follows Faraday/Ohmic coupling:

        dB/dt -> E_ind ~ (L/2) dB/dt
        J ~ sigma E_ind
        P_ohm ~ sigma E_ind^2

    All defaults are dimensionless candidate parameters. They are not calibrated
    field strengths, conductivities, orbital frequencies, or ocean dimensions.
    """

    field_bias: float = 1.0
    primary_amplitude: float = 0.35
    primary_angular_frequency: float = 0.22
    secondary_amplitude: float = 0.10
    secondary_angular_frequency: float = 0.035
    phase_offset: float = 0.0
    conductivity_proxy: float = 0.50
    loop_scale_proxy: float = 1.0
    energy_coupling_gain: float = 0.025

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in (
            "field_bias",
            "primary_amplitude",
            "primary_angular_frequency",
            "secondary_amplitude",
            "secondary_angular_frequency",
            "conductivity_proxy",
            "loop_scale_proxy",
            "energy_coupling_gain",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class MagnetoInductionObservation:
    mode: str
    time: float
    field_proxy: float
    field_derivative_proxy: float
    induced_electric_proxy: float
    current_density_proxy: float
    ohmic_power_density_proxy: float
    energy_source_rate_proxy: float
    physical_binding: str = PHYSICAL_BINDING
    parameter_status: str = PARAMETER_STATUS


def validate_magneto_induction_mode(mode: str) -> str:
    value = str(mode)
    if value not in MAGNETO_INDUCTION_MODES:
        raise ValueError(
            f"unknown magneto-induction mode {value!r}; "
            f"expected one of {MAGNETO_INDUCTION_MODES!r}"
        )
    return value


def magneto_induction_observation(
    time: float,
    *,
    mode: str = MAGNETO_INDUCTION_OFF,
    parameters: MagnetoInductionParameters | None = None,
) -> MagnetoInductionObservation:
    """Evaluate the normalized induction driver at one model time.

    A static magnetic field alone contributes no induction source here. Only
    the time derivative couples through the candidate conductivity proxy.
    """
    mode = validate_magneto_induction_mode(mode)
    p = parameters or MagnetoInductionParameters()
    p.validate()
    t = float(time)
    if not math.isfinite(t):
        raise ValueError("time must be finite")

    if mode == MAGNETO_INDUCTION_OFF:
        return MagnetoInductionObservation(
            mode=mode,
            time=t,
            field_proxy=0.0,
            field_derivative_proxy=0.0,
            induced_electric_proxy=0.0,
            current_density_proxy=0.0,
            ohmic_power_density_proxy=0.0,
            energy_source_rate_proxy=0.0,
        )

    phase1 = p.primary_angular_frequency * t + p.phase_offset
    phase2 = p.secondary_angular_frequency * t + p.phase_offset

    field = (
        p.field_bias
        + p.primary_amplitude * math.sin(phase1)
        + p.secondary_amplitude * math.sin(phase2)
    )
    dfield_dt = (
        p.primary_amplitude
        * p.primary_angular_frequency
        * math.cos(phase1)
        + p.secondary_amplitude
        * p.secondary_angular_frequency
        * math.cos(phase2)
    )

    induced_electric = 0.5 * p.loop_scale_proxy * dfield_dt
    current_density = p.conductivity_proxy * induced_electric
    ohmic_power_density = p.conductivity_proxy * induced_electric * induced_electric
    energy_source_rate = p.energy_coupling_gain * ohmic_power_density

    return MagnetoInductionObservation(
        mode=mode,
        time=t,
        field_proxy=float(field),
        field_derivative_proxy=float(dfield_dt),
        induced_electric_proxy=float(induced_electric),
        current_density_proxy=float(current_density),
        ohmic_power_density_proxy=float(ohmic_power_density),
        energy_source_rate_proxy=float(energy_source_rate),
    )
