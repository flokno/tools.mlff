#! /usr/bin/env python3

import json
import shlex
import subprocess as sp
from pathlib import Path
from typing import List

import typer
from rich import print as echo


def get_tag(file):
    """extract input parameters"""
    if file.exists():
        data_inputs = json.load(open(file))

        _keys = ("r_cut", "L", "we", "wf", "ws", "loss_variance_scaling")
        (rc, LL, we, wf, ws, vs) = (data_inputs[k] for k in _keys)

        if ws is None:
            ws = 0

        tag = f"rcut {rc:3.1f} L= {LL:d}, we: {we:.3f}, wf: {wf:.3f}, ws: {ws:.3f}"
        tag += f", loss_variance_scaling: {vs}"
    else:
        tag = str(file.parent)

    return tag


app = typer.Typer(pretty_exceptions_show_locals=False)


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

        tag = get_tag(folder / file_inputs)

        echo("... Tag:")
        echo(tag)

        file = folder / file_plot

        if not file.exists():
            echo(f"... {file} not found, skip")
            continue

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

    # check if files actually exist
    for _file in _outfiles:
        assert Path(_file).exists(), f"{_file} does not exist, arguments are folders!"

    echo("... run:")
    echo(_cmd)
    sp.run(shlex.split(_cmd))

    # cleanup
    for _file in _outfiles:
        Path(_file).unlink()

    echo("done.")


if __name__ == "__main__":
    app()
