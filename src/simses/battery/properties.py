from dataclasses import dataclass


@dataclass
class ElectricalCellProperties:
    """Electrical parameters of a single cell.

    Attributes:
        nominal_capacity: Nominal capacity in Ah.
        nominal_voltage: Nominal voltage in V.
        min_voltage: Minimum allowed terminal voltage in V.
        max_voltage: Maximum allowed terminal voltage in V.
        max_charge_rate: Maximum charge C-rate in 1/h.
        max_discharge_rate: Maximum discharge C-rate in 1/h.
        coulomb_efficiency: Coulomb efficiency in p.u. Default: 1.0.
        charge_derate_voltage_start: Terminal voltage at which charge
            current derating begins, in V. Current is linearly reduced
            from the C-rate limit at this voltage down to 0 at
            ``max_voltage``. ``None`` disables derating (default).
        discharge_derate_voltage_start: Terminal voltage at which
            discharge current derating begins, in V. Current is linearly
            reduced from the C-rate limit at this voltage down to 0 at
            ``min_voltage``. ``None`` disables derating (default).
    """

    nominal_capacity: float
    nominal_voltage: float
    min_voltage: float
    max_voltage: float
    max_charge_rate: float
    max_discharge_rate: float
    coulomb_efficiency: float = 1.0
    charge_derate_voltage_start: float | None = None
    discharge_derate_voltage_start: float | None = None


@dataclass
class ThermalCellProperties:
    """Thermal parameters of a single cell.

    Attributes:
        min_temperature: Minimum allowed cell temperature in °C.
        max_temperature: Maximum allowed cell temperature in °C.
        mass: Mass of one cell in kg.
        specific_heat: Specific heat capacity in J/kgK.
        convection_coefficient: Convective heat transfer coefficient
            between the cell surface and the thermal environment, in
            W/m²K.
    """

    min_temperature: float
    max_temperature: float
    mass: float
    specific_heat: float
    convection_coefficient: float
