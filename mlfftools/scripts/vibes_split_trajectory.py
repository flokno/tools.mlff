#! /usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import typer
from rich import print as echo

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path,
    test_split: float = 0.05,
    seed: int = None,
    outfile_train: Path = "trajectory_train.nc",
    outfile_test: Path = "trajectory_test.nc",
    outfile_splits: Path = "splits.json",
    force: bool = False,
):
    """Split samples in trajectory FILE into 2 parts for training and testing"""
    from vibes.trajectory import Trajectory

    echo(f"Read {file}")
    trajectory = Trajectory.read(file)

    # splits:
    if seed is None:
        seed = np.random.randint(65536, size=1)[0]

    echo(f"... seed: {seed:8d}")
    np.random.seed(seed)

    n_total = len(trajectory)
    n_test = int(np.floor(test_split * n_total))
    n_train = n_total - n_test - 1

    echo(f"... pick  {n_train:8d} samples for training")
    echo(f"... pick  {n_test:8d} samples for testing")

    idx_all = np.arange(n_total)

    idx_train = np.sort(np.random.choice(idx_all, n_train, replace=False))
    idx_test = np.array([ii for ii in idx_all if ii not in idx_train])

    md = trajectory.metadata
    trajectory_test = Trajectory([trajectory[ii] for ii in idx_test], metadata=md)
    trajectory_train = Trajectory([trajectory[ii] for ii in idx_train], metadata=md)

    echo(f"... write trainings set to {outfile_train}")
    trajectory_train.write(outfile_train)
    echo(f"... write test set to {outfile_test}")
    trajectory_test.write(outfile_test)

    data_splits = {
        "seed": int(seed),
        "idx_test": idx_test.tolist(),
        "idx_train": idx_train.tolist(),
    }

    echo(f"... write splits to {outfile_splits}")
    json.dump(data_splits, outfile_splits.open("w"))


if __name__ == "__main__":
    app()
