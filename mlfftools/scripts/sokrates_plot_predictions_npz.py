#! /usr/bin/env python3

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import typer
from rich import print as echo
from ase.units import GPa

app = typer.Typer(pretty_exceptions_show_locals=False)


def get_R2(x, y):
    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]

    std = np.std(x - y)
    r2 = 1 - (std / np.std(x)) ** 2
    return r2


def get_legend(mean: float, r2: float, rmse: float) -> str:
    return [f"Mean: {mean:.3f}\nR2:   {r2:.3f}\nRMSE*1e3: {rmse*1e3:.3f}"]


@app.command()
def main(
    file: Path,
    outfile: Path = None,
    fix_legacy_volume_scale: bool = False,
    fix_energy_mean: bool = False,
):
    """Convert trajectory files to MLFF input as npz file"""
    echo(f"Read data from {file}")

    # read data and create dictionaries
    data = np.load(file, allow_pickle=True)
    data_inputs = data["inputs"].item()
    data_target = data["targets"].item()
    data_predictions = data["predictions"].item()

    keys = ["E", "F"]
    ncols = 2
    if "stress" in data_predictions:
        keys.append("stress")
        ncols = 3

        if fix_legacy_volume_scale:
            echo("*** LEGACY: fix the target stress by dividing with volume")
            volumes = np.linalg.det(data_inputs["unit_cell"])
            data_target["stress"] /= volumes[:, None, None]

    if fix_energy_mean:
        echo("*** mean energies are substracted")
        data_target["E"] -= data_target["E"].mean()
        data_predictions["E"] -= data_predictions["E"].mean()

    fig, axs = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))

    for ii, label in enumerate(keys):
        ax = axs[ii]
        x = data_target[label].flatten()
        y = data_predictions[label].flatten()

        mask = np.isfinite(x)
        x = x[mask]
        y = y[mask]

        x_mean = x.mean()
        x -= x_mean
        y -= x_mean

        r2 = get_R2(x, y)
        rmse = (x - y)[~np.isnan(x)].std()

        echo(f" {label:7s}: Mean (train) = {x_mean:.6f}")
        echo(f" {label:7s}: R2 = {r2:7.3f} ;  RMSE * 1e3  = {1000*rmse:9.3f}")

        if label == "stress":
            echo(f" {label:7s}: R2 = {r2:7.3f} ;  RMSE (kbar) = {rmse/GPa*10:9.3f}")

        ax.scatter(x, y, s=5, marker=".", alpha=0.25)
        ax.plot(ax.get_xlim(), ax.get_xlim(), color="#313131", lw=1, zorder=-1)
        ax.legend(get_legend(x_mean, r2, rmse), loc=0, markerfirst=True, frameon=False)
        ax.set_title(label)
        ax.set_aspect(1, adjustable="datalim")

    fig.suptitle(file.name)

    if outfile is None:
        outfile = file.stem + ".png"

    echo(f"... save plot to {outfile}")
    fig.savefig(outfile)


if __name__ == "__main__":
    app()
