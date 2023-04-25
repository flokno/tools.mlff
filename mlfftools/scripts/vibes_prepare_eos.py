#! /usr/bin/env python3
import shutil
from pathlib import Path

import numpy as np
import typer
from ase.units import GPa
from rich import print as echo

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path = "geometry.in",
    file_geometry: Path = "geometry.in",
    file_relaxation: Path = "relaxation.in",
    file_relaxation_template: Path = "relaxation.in.template",
    dp: float = typer.Option(0.01, help="max. pressure difference in eV/AA^3 (+/-)"),
    Np: int = typer.Option(11, help="number of pressures steps"),
    base_folder: str = "eos/p_",
    key_pressure: str = "PRESSURE",
    key_file_geometry: str = "GEOMETRY",
):

    template = file_relaxation_template.read_text()

    pressures = np.linspace(-dp, dp, Np)

    for pressure in pressures:
        fol = Path(f"{base_folder}{1+pressure:.6f}")
        echo(f"... set up {fol} for p = {pressure:.6f} eV/AA = {pressure/GPa:.3f} GPa")
        fol.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, fol / file_geometry)
        infile = template.replace(key_file_geometry, str(file_geometry))
        infile = infile.replace(key_pressure, str(pressure))
        (fol / file_relaxation).write_text(infile)
        echo(f".. input files written to {fol}")


if __name__ == "__main__":
    app()
