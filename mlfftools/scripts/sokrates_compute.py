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
def main(
    files: List[Path],
    folder_model: str = "module",
    outfile: str = "predictions.nc",
    n_replicas: int = 1,
    skin: float = 1.0,
    tdep: bool = False,
    float32: bool = False,
    format: str = "aims",
):
    """Convert trajectory files to MLFF input as npz file"""
    import jax.numpy as jnp
    import xarray as xr
    from glp import atoms_to_system
    from glp.calculators import supercell
    from jax import jit, config
    from mlff.mdx import MLFFPotential

    if float32:
        dtype = jnp.float32
    else:
        config.update("jax_enable_x64", True)
        dtype = jnp.float64

    echo("Compute so3krates predictions for these files:")
    echo(files)

    if format == "vibes":
        from vibes.trajectory import reader

        assert len(files) == 1, "FIXME: Only 1 vibes trajectory at a time supported."

        file = files[0]

        echo(f"... try to read vibes trajectory from {file}")
        atoms_list = reader(file)
    else:
        echo(f"... parse files using `ase.io.read(..., format={format})`")
        atoms_list = [read(file, format=format) for file in files]

    atoms = atoms_list[0]

    # read first file and create atoms
    echo(f"... System: {atoms}")

    potential = MLFFPotential.create_from_ckpt_dir(
        folder_model, add_shift=True, dtype=dtype
    )

    calculator, state = supercell.calculator(
        potential,
        atoms_to_system(atoms, dtype=dtype),
        n_replicas=n_replicas,
        skin=skin,
    )
    calculate = jit(calculator.calculate)

    # predict
    n_atoms = len(atoms)
    n_samples = len(atoms_list)

    rows = []
    for atoms in track(atoms_list):

        predictions, state = calculate(atoms_to_system(atoms, dtype=dtype), state)
        assert not state.overflow, "FIXME"

        s = predictions["stress"] / atoms.get_volume()
        if tdep:
            s /= GPa  # TDEP wants GPa
            _stress = np.array([s[0, 0], s[1, 1], s[2, 2], s[2, 1], s[2, 0], s[1, 0]])
            _pressure = np.mean(_stress[:3])
        else:
            _stress = s
            _pressure = -np.trace(s) / 3

        row = {
            keys.cell: np.asarray(atoms.cell),
            keys.natoms: len(atoms),
            keys.volume: atoms.get_volume(),
            keys.positions_cartesian: atoms.positions,
            keys.positions: atoms.get_scaled_positions(),
            keys.forces: predictions["forces"],
            keys.energy_total: predictions["energy"],
            keys.energy_kinetic: 0.0,
            keys.energy_potential: predictions["energy"],
            keys.temperature: 0.0,
            keys.stress: _stress,
            keys.pressure: _pressure,
        }

        rows.append(row)

    if tdep:
        write_infiles(rows)
        write_meta(n_atoms=n_atoms, n_samples=n_samples)
    else:
        # rewrite data as DataArrays
        _keys = rows[0].keys()

        arrays = {}
        for k in _keys:
            array = np.concatenate([[row[k] for row in rows]], axis=0)
            dims = ["time"]
            if k in dimensions:
                dims.extend(dimensions[k])
            arrays[k] = xr.DataArray(array, name=k, dims=dims)

        ds = xr.Dataset(arrays)

        echo(ds)

        echo(f"... save data to {outfile}")
        ds.to_netcdf(outfile)


if __name__ == "__main__":
    app()
