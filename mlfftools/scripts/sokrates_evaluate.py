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
    me = energy_factor * (x - y).max()
    # nrmse = (x - y).std() / (x.max() - x.min())
    return {
        "R2": R2,
        "MAE": mae,
        "RMSE": rmse,
        "maxAE": me,
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


def get_xy(
    ds_train: xr.Dataset, ds_predict: xr.Dataset, label: str, verbose: bool = True
) -> np.ndarray:
    """get the data and account for NaN's"""
    x = ds_train[label].data.flatten()
    y = ds_predict[label].data.flatten()

    mask = np.isfinite(x)

    if sum(mask) > 2:
        x = x[mask]
        y = y[mask]
    else:
        x = np.zeros_like(y)

    if verbose:
        echo(f" {label:7s}: mean(x) (train) = {x.mean():.6f}")
        echo(f" {label:7s}: mean(y) (test)  = {y.mean():.6f}")

    x -= x.mean()
    y -= y.mean()

    return x, y


def make_plot(data_plot: list, title: str, outfile: str):
    kw_legend = {
        "loc": 0,
        "markerfirst": True,
        "frameon": False,
        "prop": {"family": "monospace"},
    }
    fig, axs = plt.subplots(ncols=len(data_plot), figsize=(4 * len(data_plot), 4))
    for ax, data in zip(axs, data_plot):
        x, y = data["x"], data["y"]
        label, errors = data["label"], data["errors"]
        ax.scatter(x, y, s=5, marker=".", alpha=0.25)
        ax.plot(ax.get_xlim(), ax.get_xlim(), color="#313131", lw=1, zorder=-1)
        ax.legend(get_legend(errors), **kw_legend)
        ax.set_title(label)
        ax.set_aspect(1, adjustable="datalim")

    fig.suptitle(title)
    echo(f"... save plot to {outfile}")
    fig.savefig(outfile)


@app.command()
def main(
    file_predictions: Path,
    file_reference: Path,
    labels: List[str] = ["energy_potential", "forces", "stress"],
    plot: bool = False,
    outfile_plot: Path = None,
    outfile_errors: Path = None,
    fix_energy_mean: bool = False,
    key_energy: str = "energy_potential",
):
    """Evaluate model errors for data in FILE, FILE_REFERENCE, optionally plot"""
    echo(
        f"Read predictions from `{file_predictions}`, training from `{file_reference}`"
    )

    ds_pred = xr.load_dataset(file_predictions)
    ds_train = xr.load_dataset(file_reference)

    if fix_energy_mean:
        echo("*** mean energies are substracted")
        ds_train[key_energy] -= ds_train[key_energy].mean()
        ds_pred[key_energy] -= ds_pred[key_energy].mean()

    rows = []
    data_plot = []
    for label in labels:
        x, y = get_xy(ds_train=ds_train, ds_predict=ds_pred, label=label)
        errors = get_errors(x, y)
        echo("Errors:")
        echo(json.dumps(errors, indent=1))
        rows.append(errors)

        data_plot.append({"label": label, "x": x, "y": y, "errors": errors})

    df = pd.DataFrame(rows, index=[l.rstrip("_potential") for l in labels])

    if outfile_errors is None:
        outfile_errors = file_predictions.stem + "_errors.csv"

    echo(f"... save errors to {outfile_errors}")
    df.to_csv(outfile_errors, index_label="target", float_format="%20.15f")

    if plot:
        if outfile_plot is None:
            outfile_plot = file_predictions.stem + ".png"

        make_plot(data_plot, title=file_predictions.name, outfile=outfile_plot)


if __name__ == "__main__":
    app()
