import os

import pandas as pd

from simses.battery.battery import BatteryState, CellType
from simses.battery.format import RoundCell26650
from simses.battery.properties import ElectricalCellProperties, ThermalCellProperties
from simses.degradation import DegradationModel
from simses.degradation.state import DegradationState
from simses.interpolation import interp1d_scalar, interp2d_scalar
from simses.model.degradation.sony_lfp_calendar import SonyLFPCalendarDegradation
from simses.model.degradation.sony_lfp_cyclic import SonyLFPCyclicDegradation


class SonyLFP(CellType):
    """Sony / Murata US26650FTC1 cylindrical LFP cell.

    Lithium iron phosphate cell with 3 Ah nominal capacity and 3.2 V
    nominal voltage. OCV, hysteresis voltage, and entropic coefficient
    are 1-D lookups in SOC; internal resistance is a 2-D lookup in
    ``(SOC, T)`` with separate charge and discharge tables. Ships a
    default degradation model
    (:class:`~simses.model.degradation.sony_lfp_calendar.SonyLFPCalendarDegradation`
    + :class:`~simses.model.degradation.sony_lfp_cyclic.SonyLFPCyclicDegradation`)
    that :class:`~simses.battery.battery.Battery` picks up when
    constructed with ``degradation=True``.

    Sources: SONY_US26650FTC1 Product Specification; and Naumann, Maik.
    *Techno-economic evaluation of stationary lithium-ion energy storage
    systems with special consideration of aging*. PhD Thesis, Technical
    University Munich, 2018.
    """

    def __init__(self):
        super().__init__(
            electrical=ElectricalCellProperties(
                nominal_capacity=3.0,  # Ah
                nominal_voltage=3.2,  # V
                max_voltage=3.6,  # V
                min_voltage=2.0,  # V
                max_charge_rate=1.0,  # 1/h
                max_discharge_rate=6.6,  # 1/h
                coulomb_efficiency=1.0,  # p.u.
            ),
            thermal=ThermalCellProperties(
                min_temperature=0.0,  # °C
                max_temperature=60.0,  # °C
                mass=0.07,  # kg per cell
                specific_heat=1001,  # J/kgK
                convection_coefficient=15,  # W/m2K
            ),
            cell_format=RoundCell26650(),
        )
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

        # 1-D look-up tables. Stored as plain Python lists so the scalar
        # interpolation helpers can use bisect on the raw sequence without
        # numpy boundary overhead.
        df_ocv = pd.read_csv(os.path.join(path, "CLFP_Sony_US26650_OCV.csv"))
        self._ocv_lut_soc = df_ocv["SOC"].tolist()
        self._ocv_lut_ocv = df_ocv["OCV"].tolist()

        df_hyst = pd.read_csv(os.path.join(path, "CLFP_Sony_US26650_HystV.csv"))
        self._hyst_lut_soc = df_hyst["SOC"].tolist()
        self._hyst_lut_hyst = df_hyst["HystV"].tolist()

        df_entropy = pd.read_csv(os.path.join(path, "CLFP_Sony_US26650_entropy.csv"))
        self._entropy_lut_soc = df_entropy["SOC"].tolist()
        self._entropy_lut_entropy = df_entropy["S"].tolist()

        # 2-D internal-resistance look-up tables (charge / discharge).
        df_rint = pd.read_csv(os.path.join(path, "CLFP_Sony_US26650_Rint.csv"))
        self._rint_lut_soc = df_rint["SOC"].tolist()
        self._rint_lut_T = df_rint["Temp"].dropna().tolist()
        self._rint_ch_mat = df_rint.iloc[:, 2:6].values.tolist()
        self._rint_dch_mat = df_rint.iloc[:, 6:].values.tolist()

    def open_circuit_voltage(self, state: BatteryState) -> float:
        return interp1d_scalar(state.soc, self._ocv_lut_soc, self._ocv_lut_ocv)

    def hysteresis_voltage(self, state: BatteryState) -> float:
        if state.is_charge:
            return 0.5 * interp1d_scalar(state.soc, self._hyst_lut_soc, self._hyst_lut_hyst)
        else:
            return -0.5 * interp1d_scalar(state.soc, self._hyst_lut_soc, self._hyst_lut_hyst)

    def entropic_coefficient(self, state: BatteryState) -> float:
        return interp1d_scalar(state.soc, self._entropy_lut_soc, self._entropy_lut_entropy)

    def internal_resistance(self, state: BatteryState) -> float:
        mat = self._rint_ch_mat if state.is_charge else self._rint_dch_mat
        return interp2d_scalar(state.soc, state.T, self._rint_lut_soc, self._rint_lut_T, mat)

    @classmethod
    def default_degradation_model(
        cls,
        initial_soc: float,
        initial_state: DegradationState | None = None,
    ) -> DegradationModel:
        return DegradationModel(
            cyclic=SonyLFPCyclicDegradation(),
            calendar=SonyLFPCalendarDegradation(),
            initial_soc=initial_soc,
            initial_state=initial_state,
        )
