#! /usr/bin/env python3

from pathlib import Path
from typing import List
import json

from ase import Atoms
import numpy as np
import typer
import xarray as xr
from rich import print as echo

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    files: List[Path],
    outfile: Path = "mlff_data.npz",
    imin: int = 0,
    imax: int = None,
    stride: int = 1,
    dropna: bool = False,
):
    """Convert trajectory files to MLFF input as npz file"""
    echo(files)

    # empty files
    _stress = []
    _z = []
    _pbc = []
    _E = []
    _F = []
    _R = []
    _velocity = []
    _unit_cell = []

    for file in files:
        with xr.open_dataset(file) as ds:

            if dropna:
                mask = np.isfinite(ds.stress.data[:, 0, 0])
            else:
                mask = np.isfinite(ds.positions.data[:, 0, 0])

            atoms = Atoms(**json.loads(ds.atoms_reference))
            nsteps = len(ds.positions[mask])

            _stress.append(ds.stress.data[mask])
            _z.append(np.tile(atoms.numbers, [nsteps, 1]))
            _pbc.append(np.tile(atoms.pbc, [nsteps, 1]))
            _E.append(ds.energy_potential.data[mask])
            _F.append(ds.forces.data[mask])
            _R.append(ds.positions.data[mask])
            _velocity.append(ds.velocities.data[mask])
            _unit_cell.append(ds.cell.data[mask])

    _stress = np.concatenate(_stress)
    _z = np.concatenate(_z)
    _pbc = np.concatenate(_pbc)
    _E = np.concatenate(_E).reshape([-1, 1])
    _F = np.concatenate(_F)
    _R = np.concatenate(_R)
    _velocity = np.concatenate(_velocity)
    _unit_cell = np.concatenate(_unit_cell)

    if imax is None:
        imax = len(_stress)

    echo(f"... pick sample {imin} to {imax} with stride {stride}")

    # prepate data
    data = {
        "z": _z[imin:imax:stride],
        "R": _R[imin:imax:stride],
        "E": _E[imin:imax:stride],
        "F": _F[imin:imax:stride],
        "stress": _stress[imin:imax:stride],
        "pbc": _pbc[imin:imax:stride],
        "unit_cell": _unit_cell[imin:imax:stride],
    }

    echo(f"... dump {len(data['z'])} samples to {outfile}")
    np.savez(outfile, **data)


if __name__ == "__main__":
    app()
