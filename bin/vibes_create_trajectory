#! /usr/bin/env python3

from pathlib import Path
from typing import List

import typer
from ase.io import read
from rich import print as echo
from vibes.trajectory import Trajectory

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    files: List[Path],
    primitive: Path = None,
    supercell: Path = None,
    outfile: Path = "trajectory.nc",
    format_geometry="aims",
    format_output="aims-output",
    force: bool = False,
):
    """Convert calculations in FILES to vibes trajectory"""
    # check if output file exists
    if Path(outfile).exists():
        echo(f"*** output file `{outfile}` exists")
        if not force:
            return
        echo(f"*** `--force` used, `{outfile}` will be overwritten")

    # read data from files
    echo(f"Parse {len(files)} file(s)")
    rows = []
    for ii, file in enumerate(files):
        echo(f"... parse file {ii+1:3d}: {str(file)}")

        try:
            atoms_list = read(file, ":", format=format_output)
        except (ValueError, IndexError):
            echo(f"*** problem in file {file}, SKIP.")
            continue

        for atoms in atoms_list:
            rows.append(atoms)

    n_samples = len(rows)
    echo(f"... found {n_samples} samples")

    if n_samples < 1:
        echo("... no data found, abort.")
        return

    trajectory = Trajectory(rows)

    # parse reference structures
    if primitive is not None:
        if supercell is not None:
            echo("Read reference structures:")
            echo(f"... primitive cell from {primitive}")
            echo(f"...      supercell from {supercell}")
            atoms_primitive = read(primitive, format=format_geometry)
            atoms_supercell = read(supercell, format=format_geometry)
            trajectory.primitive = atoms_primitive
            trajectory.supercell = atoms_supercell
        else:
            echo("*** please provide both `unitcell` and `supercell` for reference")
            echo("*** No reference structure will be added")
    else:
        echo("*** No reference structure will be added")

    echo(f"--> write trajectory to {outfile}")
    trajectory.write(outfile)


if __name__ == "__main__":
    app()
