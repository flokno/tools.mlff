#! /usr/bin/env python3

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
import xarray as xr
from matplotlib import pyplot as plt
from rich import print as echo

app = typer.Typer(pretty_exceptions_show_locals=False)


def get_R2(x, y):
    std = np.std(x - y)
    r2 = 1 - (std / np.std(x)) ** 2
    return r2


def get_errors(x, y, energy_factor=1000):
    R2 = get_R2(x, y)
    mae = energy_factor * abs(x - y).mean()
    std = energy_factor * x.std()
    rmse = energy_factor * (x - y).std()
    # nrmse = (x - y).std() / (x.max() - x.min())
    return {
        "R2": R2,
        "MAE": mae,
        "RMSE": rmse,
        "STD": std,
        "MAE/STD": mae / std,
        "RMSE/STD": rmse / std,
        # "NRMSE": nrmse,
    }


def get_legend(errors: dict) -> str:
    rows = [
        f"R2:       {errors['R2']:6.3f}",
        f"MAE*1e3:  {errors['MAE']:6.3f}",
        f"RMSE*1e3: {errors['RMSE']:6.3f}",
        f"RMSE/STD: {errors['RMSE/STD']:6.3f}",
    ]
    return ["\n".join(rows)]


@app.command()
def main(
    file: Path,
    file_training: Path,
    labels: List[str] = ["energy_potential", "forces", "stress"],
    outfile: Path = None,
    outfile_errors: Path = None,
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

    rows = []
    for ii, label in enumerate(labels):
        ax = axs[ii]
        x = ds_train[label].data.flatten()
        y = ds_pred[label].data.flatten()

        mask = np.isfinite(x)
        x = x[mask]
        y = y[mask]

        echo(f" {label:7s}: mean(x) (train) = {x.mean():.6f}")
        echo(f" {label:7s}: mean(y) (test)  = {y.mean():.6f}")

        x -= x.mean()
        y -= y.mean()

        errors = get_errors(x, y)
        echo("Errors:")
        echo(json.dumps(errors, indent=1))
        rows.append(errors)

        ax.scatter(x, y, s=5, marker=".", alpha=0.25)
        ax.plot(ax.get_xlim(), ax.get_xlim(), color="#313131", lw=1, zorder=-1)
        kw = {
            "loc": 0,
            "markerfirst": True,
            "frameon": False,
            "prop": {"family": "monospace"},
        }
        ax.legend(get_legend(errors), **kw)
        ax.set_title(label)
        ax.set_aspect(1, adjustable="datalim")

    df = pd.DataFrame(rows, index=[l.rstrip("_potential") for l in labels])

    fig.suptitle(file.name)

    if outfile is None:
        outfile = file.stem + ".png"

    echo(f"... save plot to {outfile}")
    fig.savefig(outfile)

    if outfile_errors is None:
        outfile_errors = file.stem + "_errors.csv"

    echo(f"... save errors to {outfile_errors}")
    df.to_csv(outfile_errors, index_label="target", float_format="%20.15f")


if __name__ == "__main__":
    app()
