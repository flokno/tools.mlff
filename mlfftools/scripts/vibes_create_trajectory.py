#! /usr/bin/env python3

from pathlib import Path
from typing import List

import typer
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from rich import print as echo
from vibes.trajectory import Trajectory


def read_file(file: Path, format: str = None) -> list:
    """Read FILE and turn into list of Atoms objects"""
    if format is None:
        if "aims.out" in str(file):
            format = "aims-output"
        if file.suffix == ".npz":
            format = "mlff"

    if format == "mlff":
        import numpy as np
        from mlff.properties import property_names as pn
        from mlff.properties import md17_property_keys as prop_keys

        with np.load(file, allow_pickle=True) as data:
            data_dict = dict(data)

        Rs = data_dict[prop_keys[pn.atomic_position]]
        Zs = data_dict[prop_keys[pn.atomic_type]]
        Cs = data_dict[prop_keys[pn.unit_cell]]
        PBCs = data_dict[prop_keys[pn.pbc]]
        Es = data_dict[prop_keys[pn.energy]].squeeze()
        Fs = data_dict[prop_keys[pn.force]]
        Ss = data_dict[prop_keys[pn.stress]]

        atoms_list = []
        for (R, Z, C, PBC, E, F, S) in zip(Rs, Zs, Cs, PBCs, Es, Fs, Ss):
            atoms = Atoms(positions=R, numbers=Z, cell=C, pbc=PBC)
            # results = {"energy": E, "forces": F, "stress": S}
            atoms.calc = SinglePointCalculator(atoms, energy=E, forces=F, stress=S)
            atoms_list.append(atoms)
    else:
        atoms_list = read(file, ":", format=format)

    return atoms_list


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    files: List[Path],
    primitive: Path = None,
    supercell: Path = None,
    outfile: Path = "trajectory.nc",
    format_geometry="aims",
    format_output=None,
    stride: int = 1,
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
            atoms_list = read_file(file, format=format_output)
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

    if stride > 1:
        echo(f"... use a stride of {stride}")
        rows = rows[::stride]
        echo(f"... reduce no. of samples from {n_samples} to {len(rows)}")

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
