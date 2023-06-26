#! /usr/bin/env python3

import json
import shlex
import subprocess as sp
from pathlib import Path
from typing import List

import typer
from rich import print as echo

app = typer.Typer()


@app.command()
def main(
    folders: List[Path],
    file_plot: str = "plot_test.png",
    file_inputs: str = "inputs.json",
    outfile: Path = "sokrates_model_gallery.pdf",
    outfile_prefix: str = "plot_",
    outfile_suffix: str = "pdf",
    splice: int = 50,
    pointsize: int = 20,
):

    _outfiles = []
    for ii, folder in enumerate(folders):
        data_inputs = json.load(open(folder / file_inputs))

        _keys = ("r_cut", "L", "we", "wf", "ws", "loss_variance_scaling")
        (rc, LL, we, wf, ws, vs) = (data_inputs[k] for k in _keys)

        if ws is None:
            ws = 0

        tag = f"rcut {rc:3.1f} L= {LL:d}, we: {we:.3f}, wf: {wf:.3f}, ws: {ws:.3f}"
        tag += f", loss_variance_scaling: {vs}"

        echo(tag)

        file = folder / file_plot
        _outfile = f"{outfile_prefix}{ii:03d}.{outfile_suffix}"

        cmd = f"convert {file} "
        cmd += "-gravity South "
        cmd += f"-pointsize {pointsize} "
        cmd += f"-splice 0x{splice:d} "
        cmd += f"-annotate +0+20 '{tag}' "
        cmd += f"{_outfile}"

        echo("... run:")
        echo(cmd)

        sp.run(shlex.split(cmd))
        _outfiles.append(_outfile)

    _cmd = "pdfjam "
    for _outfile in _outfiles:
        _cmd += f"{_outfile} '-' "
    _cmd += "--landscape "
    _cmd += f"--outfile {outfile}"

    echo("... run:")
    echo(_cmd)
    sp.run(shlex.split(_cmd))

    # cleanup
    for _file in _outfiles:
        Path(_file).unlink()

    echo("done.")


if __name__ == "__main__":
    app()
