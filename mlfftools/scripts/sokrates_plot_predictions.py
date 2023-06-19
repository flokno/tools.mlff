#! /usr/bin/env python3

from pathlib import Path
from typing import List

import typer

from mlfftools.scripts.sokrates_evaluate import main as _main

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path,
    file_training: Path,
    labels: List[str] = ["energy_potential", "forces", "stress"],
    outfile: Path = None,
    outfile_errors: Path = None,
    fix_energy_mean: bool = False,
    key_energy: str = "energy_potential",
):
    """(legacy) ML plot for files (training, prediction)"""

    _main(
        file=file,
        file_training=file_training,
        labels=labels,
        plot=True,
        outfile_plot=outfile,
        outfile_errors=outfile_errors,
        fix_energy_mean=fix_energy_mean,
        key_energy=key_energy,
    )


if __name__ == "__main__":
    app()
