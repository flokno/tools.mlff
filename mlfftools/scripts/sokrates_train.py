import json
import logging
from pathlib import Path

import typer
from mlff.properties import md17_property_keys as prop_keys
from rich import print as echo

logging.basicConfig(level=logging.INFO)


def get_so3krates_net(r_cut, L, F, l_min, l_max, mic):
    """Prepare So3krates net"""
    from mlff.nn import So3krates

    degrees = list(range(l_min, l_max + 1))
    net = So3krates(
        prop_keys=prop_keys,
        F=F,
        n_layer=L,
        geometry_embed_kwargs={
            "degrees": degrees,
            "mic": mic,
            "r_cut": r_cut,
        },
        so3krates_layer_kwargs={"degrees": degrees},
    )

    return net


def get_so3kratACE_net(
    r_cut, L, F, l_min, l_max, mic, atomic_types, max_body_order, F_body_order
):
    """Prepare So3kratACE net"""
    from mlff.nn import So3kratACE

    degrees = list(range(l_min, l_max + 1))
    net = So3kratACE(
        prop_keys=prop_keys,
        F=F,
        n_layer=L,
        atomic_types=atomic_types,
        geometry_embed_kwargs={"degrees": degrees, "mic": mic, "r_cut": r_cut},
        so3kratace_layer_kwargs={
            "degrees": degrees,
            "max_body_order": max_body_order,
            "bo_features": F_body_order,
        },
    )

    return net


def log_params(params: dict, outfile_inputs: str):
    echo("We use the following settings:")
    echo(params)
    echo(f"... write input arguments to {outfile_inputs}")
    with open(outfile_inputs, "w") as f:
        json.dump(params, f, indent=1)


app = typer.Typer(pretty_exceptions_show_locals=False)

_we_default = typer.Option(0.01, "--weight-energy", "-we")
_wf_default = typer.Option(1.0, "--weight-forces", "-wf")
_ws_default = typer.Option(None, "--weight-stress", "-ws")


@app.command()
def train_so3krates(
    ctx: typer.Context,
    file_data: Path,
    ckpt_dir: Path = "module",
    r_cut: float = 5.0,
    L: int = 3,
    F: int = 132,
    l_min: int = 1,
    l_max: int = 3,
    max_body_order: int = 2,
    F_body_order: int = 1,
    we: float = _we_default,
    wf: float = _wf_default,
    ws: float = _ws_default,
    loss_variance_scaling: bool = False,
    epochs: int = 2000,
    train_split: float = 0.8,
    eval_every_t: int = None,
    mic: bool = True,
    float64: bool = False,
    lr: float = 1e-3,
    lr_stop: float = 1e-5,
    lr_decay_exp_transition_steps: int = 100000,
    lr_decay_exp_decay_factor: float = 0.7,
    clip_by_global_norm: float = None,
    shift_mean: bool = True,
    size_batch: int = None,
    size_batch_training: int = None,
    size_batch_validation: int = None,
    seed_model: int = 0,
    seed_data: int = 0,
    seed_training: int = 0,
    wandb_name: str = None,
    wandb_group: str = None,
    wandb_project: str = None,
    outfile_inputs: Path = "inputs.json",
    overwrite_module: bool = False,
    ace: bool = False,
):
    from mlfftools.train import prepare_run, get_dataset_and_n_train

    if float64:
        from jax.config import config

        config.update("jax_enable_x64", True)

    # state and save settings
    log_params(ctx.params, outfile_inputs)

    # read data, necessary only for ACE
    n_train, data_set = get_dataset_and_n_train(**ctx.params)

    # create the net
    kw = {"r_cut": r_cut, "L": L, "F": F, "l_min": l_min, "l_max": l_max, "mic": mic}
    if ace:
        echo("... let's ACE!")
        net = get_so3kratACE_net(
            **kw,
            atomic_types=data_set.all_atomic_types(),
            max_body_order=max_body_order,
            F_body_order=F_body_order,
        )
    else:
        net = get_so3krates_net(**kw)

    # create run
    coach, coach_run_kwargs = prepare_run(net, data_set, n_train, **ctx.params)

    echo("... go 🚀")
    coach.run(**coach_run_kwargs)


if __name__ == "__main__":
    app()
