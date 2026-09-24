import numpy as np
import pandas as pd

from .common import interp_clamped

CHARGING_LINE: int = 11
DISCHARGING_LINE: int = 14


def calc_ocv(measurements: dict[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:

    charge_results = []
    discharge_results = []

    for file, measurement in measurements.items():
        charge_mask = measurement["Line"] == CHARGING_LINE
        discharge_mask = measurement["Line"] == DISCHARGING_LINE

        ocv_charge = measurement.loc[charge_mask, ["SOC", "U[V]"]].dropna().copy()
        ocv_charge["File"] = file

        ocv_discharge = measurement.loc[discharge_mask, ["SOC", "U[V]"]].dropna().copy()
        ocv_discharge["File"] = file

        charge_results.append(ocv_charge)
        discharge_results.append(ocv_discharge)

    charge_df = pd.concat(charge_results, ignore_index=True)
    discharge_df = pd.concat(discharge_results, ignore_index=True)

    return charge_df, discharge_df


def build_ocv_lut(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    soc_start: float = 0.0,
    soc_stop: float = 1.0,
    soc_step: float = 0.01,
    truncation_percent: float = 5.0,
) -> dict[str, pd.DataFrame]:

    soc_lut = np.arange(soc_start, soc_stop + soc_step, soc_step).tolist()

    lut_results = {}

    files = set(charge_df["File"]) & set(discharge_df["File"])

    for file in files:
        charge = charge_df.loc[charge_df["File"] == file].sort_values("SOC")

        discharge = discharge_df.loc[discharge_df["File"] == file].sort_values("SOC")

        soc_charge = charge["SOC"].tolist()
        v_charge = charge["U[V]"].tolist()

        soc_discharge = discharge["SOC"].tolist()
        v_discharge = discharge["U[V]"].tolist()

        ocv_lut = []
        hys_lut = []

        for soc in soc_lut:
            v_ch = interp_clamped(soc, soc_charge, v_charge)

            v_dch = interp_clamped(soc, soc_discharge, v_discharge)

            ocv_lut.append((v_ch + v_dch) / 2)

            hys_lut.append((v_ch - v_dch) / 2)

        n_trunc = round(len(soc_lut) * truncation_percent / 100)

        if n_trunc > 0:
            hys_lut[:n_trunc] = [hys_lut[n_trunc]] * n_trunc
            hys_lut[-n_trunc:] = [hys_lut[-n_trunc - 1]] * n_trunc

        lut_results[file] = pd.DataFrame({
            "SOC": soc_lut,
            "OCV[V]": ocv_lut,
            "Vhyst[V]": hys_lut,
        })

    return lut_results
