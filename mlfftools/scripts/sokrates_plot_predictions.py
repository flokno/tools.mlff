#! /usr/bin/env python3

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import typer
from rich import print as echo
from ase.units import GPa
from typing import List
import xarray as xr

app = typer.Typer(pretty_exceptions_show_locals=False)


def get_R2(x, y):
    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]

    std = np.std(x - y)
    r2 = 1 - (std / np.std(x)) ** 2
    return r2


def get_legend(x_mean: float, y_mean: float, r2: float, rmse: float) -> str:
    rows = [
        f"Mean x: {x_mean:.3f}",
        f"Mean y: {y_mean:.3f}",
        f"R2:   {r2:.3f}",
        f"RMSE*1e3: {rmse*1e3:.3f}",
    ]
    return ["\n".join(rows)]


@app.command()
def main(
    file: Path,
    file_training: Path,
    labels: List[str] = ["energy_potential", "forces", "stress"],
    outfile: Path = None,
    fix_energy_mean: bool = False,
):
    """ML plot for files (training, prediction)"""
    echo(f"Read predictions from `{file}`, training from `{file_training}`")

    ds_pred = xr.load_dataset(file)
    ds_train = xr.load_dataset(file_training)

    if fix_energy_mean:
        echo("*** mean energies are substracted")
        ds_train["energy_potential"] -= ds_train["energy_potential"].mean()
        ds_pred["energy_potential"] -= ds_pred["energy_potential"].mean()

    ncols = len(labels)
    fig, axs = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))

    for ii, label in enumerate(labels):
        ax = axs[ii]
        x = ds_train[label].data.flatten()
        y = ds_pred[label].data.flatten()

        mask = np.isfinite(x)
        x = x[mask]
        y = y[mask]

        x_mean = x.mean()
        y_mean = y.mean()
        x -= x_mean
        y -= y_mean

        r2 = get_R2(x, y)
        rmse = (x - y)[~np.isnan(x)].std()

        echo(f" {label:7s}: x_mean (train) = {x_mean:.6f}")
        echo(f" {label:7s}: y_mean (test)  = {y_mean:.6f}")
        echo(f" {label:7s}: R2 = {r2:7.3f} ;  RMSE * 1e3  = {1000*rmse:9.3f}")

        if label == "stress":
            echo(f" {label:7s}: R2 = {r2:7.3f} ;  RMSE (kbar) = {rmse/GPa*10:9.3f}")

        ax.scatter(x, y, s=5, marker=".", alpha=0.25)
        ax.plot(ax.get_xlim(), ax.get_xlim(), color="#313131", lw=1, zorder=-1)
        kw = {"loc": 0, "markerfirst": True, "frameon": False}
        ax.legend(get_legend(x_mean, y_mean, r2, rmse), **kw)
        ax.set_title(label)
        ax.set_aspect(1, adjustable="datalim")

    fig.suptitle(file.name)

    if outfile is None:
        outfile = file.stem + ".png"

    echo(f"... save plot to {outfile}")
    fig.savefig(outfile)


if __name__ == "__main__":
    app()
