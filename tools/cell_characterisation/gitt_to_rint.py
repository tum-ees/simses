import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import interp_clamped

GITT_COLUMNS: tuple[str, ...] = ("~Time[s]", "Line", "U[V]", "I[A]", "Ah[Ah]", "Cyc-Count")

MEASUREMENT_LINES: tuple[int, ...] = (22, 33, 45, 56, 68, 79)
VALID_LINES: tuple[int, ...] = (45, 56, 68, 79)
DEFAULT_SOC_GRID: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 101))
DEFAULT_PULSES: tuple[int, ...] = tuple(range(1, 21))

def _calc_rint_helper(
    measurement: pd.DataFrame, file: str, pulse_edge_lines: tuple[int, ...] = MEASUREMENT_LINES, pulses: tuple[int, ...] = DEFAULT_PULSES
):
    events = measurement[measurement["Line"].isin(pulse_edge_lines) & measurement["Cyc-Count"].isin(pulses)].copy()

    events["Sample"] = events.groupby(["Line", "Cyc-Count"]).cumcount()
    edges = events[events["Sample"] < 2]

    edge = edges.set_index(["Line", "Cyc-Count", "Sample"])[["U[V]", "I[A]", "SOC"]].unstack("Sample")

    rint = pd.DataFrame(index=edge.index)
    rint["Rint[Ohm]"] = (edge["U[V]", 1] - edge["U[V]", 0]) / (edge["I[A]", 1] - edge["I[A]", 0])
    rint["SOC"] = edge["SOC", 0]
    rint["File"] = file

    return rint.reset_index()


def calc_rint(
    measurements: dict[pd.DataFrame], pulse_edge_lines: tuple[int, ...] = MEASUREMENT_LINES, pulses: tuple[int, ...] = DEFAULT_PULSES
) -> pd.DataFrame:
    return pd.concat(
        [
            _calc_rint_helper(measurement, file, pulse_edge_lines, pulses)
            for file, measurement in measurements.items()
        ],
        ignore_index=True,
    )


def build_rint_lut(
    result: pd.DataFrame,
    soc_grid: tuple[float, ...] = DEFAULT_SOC_GRID,
    valid_lines: tuple[int, ...] = VALID_LINES,
) -> pd.DataFrame:

    for_lut = (
        result.loc[result["Line"].isin(valid_lines)]
        .dropna(subset=["SOC", "Rint[Ohm]"])
        .copy()
    )

    interpolated_rints = []

    for (_file, _line), group in for_lut.groupby(["File", "Line"]):

        group = group.sort_values("SOC")

        group = (
            group.groupby("SOC", as_index=False)["Rint[Ohm]"]
            .mean()
            .sort_values("SOC")
        )

        socs = group["SOC"].to_numpy()
        rints = group["Rint[Ohm]"].to_numpy()

        if len(socs) < 2:
            continue

        assert np.all(np.diff(socs) > 0)

        curve = [
            interp_clamped(soc, socs, rints)
            for soc in soc_grid
        ]

        interpolated_rints.append(curve)

    if not interpolated_rints:
        raise ValueError("No valid Rint curves found.")

    # (n_curves, n_soc_points)
    interpolated_rints = np.asarray(interpolated_rints)

    # mean across files/lines at every LUT SOC
    mean_rint = np.mean(interpolated_rints, axis=0)

    return pd.DataFrame({
        "SOC": soc_grid,
        "Rint[Ohm]": mean_rint,
    })

def plot_lut(
        result: pd.DataFrame,
        lut: pd.DataFrame
) -> None:
    groups = result.groupby("File")

    for file, group in groups:
        plt.plot(
            group["SOC"],
            group["Rint[Ohm]"],
            label=file,
        )

    plt.plot(lut["SOC"], lut["Rint[Ohm]"], label="LUT", linewidth=3, c="red")
    plt.ylabel("Rint[Ohm]")
    plt.xlabel("SOC")

    plt.legend()
