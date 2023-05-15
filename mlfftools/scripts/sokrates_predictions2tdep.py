#! /usr/bin/env python3

from pathlib import Path
from typing import List

import numpy as np
import typer
from ase.io import read
from ase.units import GPa
from rich import print as echo
from rich.progress import track
from tdeptools.scripts.tdep_parse_output import (
    dimensions,
    keys,
    write_infiles,
    write_meta,
)

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(file: Path):
    """Convert trajectory files to MLFF input as npz file"""
    import xarray as xr

    echo(f"Convert predictions in {file} to TDEP input")

    ds = xr.load_dataset(file)

    echo(ds)
    rows = []
    for ii, time in enumerate(track(ds["time"])):

        ds_step = ds.sel(time=time)  # dataset for this step

        s = ds_step[keys.stress].data / GPa
        _stress = np.array([s[0, 0], s[1, 1], s[2, 2], s[2, 1], s[2, 0], s[1, 0]])
        _pressure = ds_step[keys.pressure].data.squeeze() / GPa

        row = {key: ds_step[key].data.squeeze() for key in ds_step.data_vars}
        row[keys.stress] = _stress
        row[keys.pressure] = _pressure
        rows.append(row)

    write_infiles(rows)
    write_meta(n_atoms=len(ds.atom), n_samples=len(ds.time))


if __name__ == "__main__":
    app()
