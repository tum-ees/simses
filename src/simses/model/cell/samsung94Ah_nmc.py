import math

from simses.battery.battery import BatteryState
from simses.battery.cell import CellType
from simses.battery.format import PrismaticCell
from simses.battery.properties import ElectricalCellProperties, ThermalCellProperties


class Samsung94AhNMC(CellType):
    """Samsung 94 Ah prismatic NMC cell.

    High-energy lithium nickel-manganese-cobalt-oxide cell typical of
    stationary and automotive applications. Analytical ``OCV(SOC)`` as a
    sum of sigmoids and a linear term; constant internal resistance
    (SoC- and temperature-independent).

    Source: Collath et al., "Suitability of late-life lithium-ion cells
    for battery energy storage systems", Journal of Energy Storage 87
    (2024) 111508, doi:10.1016/j.est.2024.111508.
    """

    def __init__(self) -> None:
        super().__init__(
            electrical=ElectricalCellProperties(
                nominal_capacity=94.0,  # Ah
                nominal_voltage=3.68,  # V
                max_voltage=4.15,  # V
                min_voltage=2.7,  # V
                max_charge_rate=2.0,  # 1/h
                max_discharge_rate=2.0,  # 1/h
                coulomb_efficiency=1.0,  # p.u.
            ),
            thermal=ThermalCellProperties(
                min_temperature=-40.0,  # °C
                max_temperature=60.0,  # °C
                mass=2.1,  # kg per cell
                specific_heat=1000,  # J/kgK
                convection_coefficient=15,  # W/m2K
            ),
            cell_format=PrismaticCell(
                height=125,  # mm
                width=45.0,  # mm
                length=173.0,  # mm
            ),
        )

    def open_circuit_voltage(self, state: BatteryState) -> float:
        a1 = 3.3479
        a2 = -6.7241
        a3 = 2.5958
        a4 = -61.9684
        b1 = 0.6350
        b2 = 1.4376
        k0 = 4.5868
        k1 = 3.1768
        k2 = -3.8418
        k3 = -4.6932
        k4 = 0.3618
        k5 = 0.9949

        soc = state.soc

        ocv = (
            k0
            + k1 / (1 + math.exp(a1 * (soc - b1)))
            + k2 / (1 + math.exp(a2 * (soc - b2)))
            + k3 / (1 + math.exp(a3 * (soc - 1)))
            + k4 / (1 + math.exp(a4 * soc))
            + k5 * soc
        )
        return ocv

    def internal_resistance(self, state: BatteryState) -> float:
        return 0.819e-3
