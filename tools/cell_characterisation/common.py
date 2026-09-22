from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from simses.interpolation import interp1d_scalar


def read_measurements(files: list[Path], cols: tuple[str, ...] | None = None) -> dict[pd.DataFrame]:
    measurements = {}

    for data_path in files:
        measurement = pd.read_csv(
            filepath_or_buffer=data_path,
            sep="\t",
            skiprows=32,
            encoding="latin1",
            usecols=cols,
        )

        Q = measurement["Ah[Ah]"].max() - measurement["Ah[Ah]"].min()
        measurement["SOC"] = ((measurement["Ah[Ah]"] - measurement["Ah[Ah]"].min()) / Q).clip(0.0, 1.0)

        measurements[data_path.stem] = measurement

    return measurements

def plot_measurements(
        measurements: dict[pd.DataFrame],
        slice_step: int = 1
) -> None:
    fig, axes = plt.subplots(3, 1, sharex=False, figsize=(10, 7))

    for file, measurement in measurements.items():
        overview = measurement.iloc[::slice_step]
        x = overview["~Time[s]"]/3600

        axes[0].plot(x, overview["U[V]"], label=file)
        axes[1].plot(x, overview["I[A]"])
        axes[2].plot(x, overview["SOC"])

    axes[0].set_ylabel("U[V]")
    axes[1].set_ylabel("I[A]")
    axes[2].set_ylabel("SOC")
    axes[2].set_xlabel("Time[h]")
    axes[0].legend()

    plt.tight_layout()

def interp_clamped(x, xp, fp):
    if x <= xp[0]:
        return fp[0]

    if x >= xp[-1]:
        return fp[-1]

    return interp1d_scalar(x, xp, fp)
