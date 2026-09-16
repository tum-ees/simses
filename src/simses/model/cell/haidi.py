import os

import pandas as pd

from simses.battery.battery import BatteryState, CellType
from simses.battery.format import RoundCell
from simses.battery.properties import ElectricalCellProperties, ThermalCellProperties
from simses.interpolation import interp1d_scalar


class Haidi(CellType):
    """HDCF18650-1800mAh-3.2V cylindrical Li-ion cell."""

    def __init__(self):
        super().__init__(
            electrical=ElectricalCellProperties(
                nominal_capacity=1.8,  # Ah
                nominal_voltage=3.2,  # V
                max_voltage=3.65,  # V
                min_voltage=2.0,  # V
                max_charge_rate=1.0,  # 1/h
                max_discharge_rate=5.0,  # 1/h
                self_discharge_rate=0.0,  # Placeholder
                coulomb_efficiency=1.0,  # p.u.; Placeholder
            ),
            thermal=ThermalCellProperties(
                # min and max_temp direction dependent on sheet!
                min_temperature=0.0,  # °C, lower bound recommended charge cell environment temperature
                max_temperature=60.0,  # °C, maximum short term allowable charge
                mass=0.045,  # kg per cell
                # specific_heat=,  # J/kgK
                # convection_coefficient=,  # W/m2K
            ),
            cell_format=RoundCell(
                diameter=18.2,  # mm
                length=65.4,  # mm
            ),
        )

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        df = pd.read_csv(os.path.join(path, "haidi_ocv_hys.csv"))

        self._ocv_lut_soc = df["SOC"].tolist()
        self._ocv_lut_ocv = df["OCV"].tolist()

        self._hyst_lut_soc = df["SOC"].tolist()
        self._hyst_lut_hyst = df["HystV"].tolist()

        df = pd.read_csv(os.path.join(path, "haidi_rint.csv"))

        self._rint_lut_soc = df["SOC"].tolist()
        self._rint_lut_rint = df["Rint[Ohm]"].tolist()

    def open_circuit_voltage(self, state: BatteryState) -> float:
        """
        LUT derived from the 124_02_pOCV_C100_LFP_Haidi_YF047 measurement data
        """
        return interp1d_scalar(state.soc, self._ocv_lut_soc, self._ocv_lut_ocv)

    def hysteresis_voltage(self, state: BatteryState) -> float:
        """
        LUT derived from the 124_02_pOCV_C100_LFP_Haidi_YF047 measurement data
        """
        return interp1d_scalar(state.soc, self._hyst_lut_soc, self._hyst_lut_hyst)

    def internal_resistance(self, state):
        """
        LUT derived from the 105_03_GITT_LFP_Haidi_YF011 measurement data, using the 1C and C/5 charge and discharge branches
        """
        return interp1d_scalar(state.soc, self._rint_lut_soc, self._rint_lut_rint)
