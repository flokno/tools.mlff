#! /usr/bin/env python3

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from ase.io import read
from ase.units import Hartree
from rich import print as echo
from rich.progress import track

eV = 27.211384500

aims_correction = eV / Hartree


app = typer.Typer(pretty_exceptions_show_locals=False)


def get_row_from_atoms(atoms, fix_codata=False) -> dict:
    """compile data in a dict"""
    energy = atoms.get_potential_energy()
    if fix_codata:
        # convert from CODATA 2014 back to 2002
        echo("*** apply CODATA correction to 2002 (=aims)")
        energy *= aims_correction

    return {
        "volume": atoms.get_volume(),
        "energy": energy,
        "pressure": -np.mean(atoms.get_stress()[:3]),
        "N": len(atoms),
    }


@app.command()
def main(
    files: List[Path],
    outfile: str = "eos.csv",
    format: str = "aims-output",
    fix_codata: bool = False,
):

    rows = []

    for file in track(files):

        echo(f"... parse {file}")

        if format == "vibes":
            from vibes.trajectory import reader

            atoms = reader(file, verbose=False)[-1]

        else:
            atoms = read(file, index=-1, format=format)
            print(atoms.get_volume())

        row = get_row_from_atoms(atoms, fix_codata=fix_codata)
        rows.append(row)

    df = pd.DataFrame(rows)

    echo(f"... write data to {outfile}")

    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    app()
