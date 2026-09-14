from simses.battery.battery import BatteryState
from simses.battery.cell import CellType
from simses.battery.format import RoundCell
from simses.battery.properties import ElectricalCellProperties, ThermalCellProperties


class SCA3200(CellType):
    # use soc limits for working range, set voltage limits at [0.0, 3.0] and use headroom for voltage over/undershoot?
    def __init__(
        self,
        capacitance: float = 3200.0,  # F
        rated_voltage: float = 2.85,  # V
        min_voltage=1.425,  # V
        internal_resistance: float = 0.18e-3,  # Ohm
        leakage_current=11e-3,  # A
        peak_power=11.3e3,  # W
        diameter: float = 60.2,  # mm
        length: float = 138.0,  # mm
        mass: float = 0.53,  # kg
        t_min: float = -40.0,  # °C
        t_max: float = 65.0,  # °C
        thermal_cap=633.7,  # J/°C
    ) -> None:

        energy_Wh = (rated_voltage**2 - min_voltage**2) * capacitance / 2 / 3600
        nom_capacity_Ah = capacitance * (rated_voltage - min_voltage) / 3600

        c_rate = peak_power / energy_Wh  # very basic approximation

        electrical = ElectricalCellProperties(
            nominal_capacity=nom_capacity_Ah,
            nominal_voltage=energy_Wh / nom_capacity_Ah,
            max_voltage=rated_voltage,
            min_voltage=min_voltage,
            max_charge_rate=c_rate,
            max_discharge_rate=c_rate,
            coulomb_efficiency=1.0,
        )

        thermal = ThermalCellProperties(
            min_temperature=t_min,
            max_temperature=t_max,
            mass=mass,
            specific_heat=thermal_cap / mass,
            convection_coefficient=15,  # W/m2K (placeholder value)
        )

        cell_format = RoundCell(diameter, length)

        super().__init__(
            electrical=electrical,
            thermal=thermal,
            cell_format=cell_format,
        )

        self.r_int = internal_resistance
        self.i_sd = leakage_current

    def open_circuit_voltage(self, state: BatteryState) -> float:
        return self.electrical.min_voltage + (self.electrical.max_voltage - self.electrical.min_voltage) * state.soc

    def internal_resistance(self, state: BatteryState) -> float:
        return self.r_int

    def self_discharge_current(self, state) -> float:
        return self.i_sd
