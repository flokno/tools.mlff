from pathlib import Path

import numpy as np
import typer
from mlff.properties import md17_property_keys as prop_keys
from rich import print as echo


prop_keys_inverse = {v: k for k, v in prop_keys.items()}


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train_so3krates(ctx: typer.Context, file: Path, verbose: bool = False):
    echo(f"Show MLFF training data in {file.absolute()}")
    # read data and create dictionaries
    with np.load(file, allow_pickle=True) as data:
        data_dict = dict(data)

        echo(f"... all keys: {set(data_dict.keys())}")

        for key in data_dict.keys():
            array = data_dict[key]
            if verbose:
                echo(f"Key: {key} -> {prop_keys_inverse[key]}")
                echo(f"... shape: {array.shape}")

        echo(f"--> number of samples: {len(array)}")


if __name__ == "__main__":
    app()
