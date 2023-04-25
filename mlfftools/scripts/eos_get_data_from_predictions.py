#! /usr/bin/env python3

from pathlib import Path

import xarray as xr
import typer
from rich import print as echo

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path = "predictions.nc",
    outfile: str = "eos.csv",
):

    ds = xr.load_dataset(file)[["volume", "energy_potential", "pressure", "natoms"]]
    ds = ds.rename({"energy_potential": "energy", "natoms": "N"})
    df = ds.to_dataframe()

    echo(f"... write data to {outfile}")
    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    app()
