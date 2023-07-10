import json
import logging
from pathlib import Path

import numpy as np
import typer
from mlff.properties import md17_property_keys as prop_keys
from mlff.properties import property_names as pn
from rich import print as echo

logging.basicConfig(level=logging.INFO)


def autoset_batch_size(u):
    if u < 500:
        return 1
    elif 500 <= u < 1000:
        return 5
    elif 1000 <= u < 10_000:
        return 10
    elif u >= 10_000:
        return 100


def get_scales_with_variance_scaling(data, targets):
    """Scale targets with variance"""
    scales = {}
    for t in targets:
        if t == pn.stress:
            scales[prop_keys[t]] = 1 / np.nanvar(data["train"][prop_keys[t]], axis=0)
        elif t == pn.energy:
            scales[prop_keys[t]] = 1 / np.nanvar(data["train"][prop_keys[t]])
        elif t == pn.force:
            force_data_train = data["train"][prop_keys[t]]
            node_msk_train = data["train"][prop_keys[pn.node_mask]]
            echo(force_data_train.shape)
            echo(node_msk_train.shape)
            scales[prop_keys[t]] = 1 / np.nanvar(force_data_train[node_msk_train])
    return scales


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
):
    import jax
    import jax.numpy as jnp
    import wandb
    from mlff.data import DataSet, DataTuple
    from mlff.io import bundle_dicts, save_dict
    from mlff.nn import So3krates
    from mlff.nn.stacknet import (
        get_energy_force_stress_fn,
        get_obs_and_force_fn,
        get_observable_fn,
    )
    from mlff.training import Coach, Optimizer, create_train_state, get_loss_fn

    if float64:
        from jax.config import config

        config.update("jax_enable_x64", True)

    # state settings
    echo("We use the following settings:")
    echo(ctx.params)

    # prepare inputs and targets
    def get_inputs(mic=False):
        inputs = [pn.atomic_type, pn.atomic_position, pn.idx_i, pn.idx_j, pn.node_mask]
        if mic:
            inputs += [pn.unit_cell, pn.cell_offset]
        return inputs

    def get_targets(stress=False):
        targets = [pn.energy, pn.force]
        if stress:
            targets += [pn.stress]
        return targets

    use_stress = ws is not None
    inputs = get_inputs(mic=mic)
    targets = get_targets(stress=use_stress)

    # data and loss
    data = dict(np.load(file_data))

    def get_dataset_scales_splits(
        data,
        stress=False,
        shift_mean=False,
        loss_variance_scaling=False,
        ckpt_dir="module",
        file_splits="splits.json",
        file_scales="scales.json",
    ):
        if stress:  # <- we want to train with stress
            cell_key = prop_keys[pn.unit_cell]
            stress_key = prop_keys[pn.stress]

            _stress = data[stress_key]
            # re-scale stress with cell volume
            cells = data[cell_key]  # shape: (B,3,3)
            cell_volumes = np.abs(np.linalg.det(cells))  # shape: (B)
            data[stress_key] = _stress * cell_volumes[:, None, None]

        # splits:
        n_total = len(data[prop_keys[pn.energy]])
        n_train = int(np.floor(train_split * n_total))
        n_valid = n_total - n_train - 1

        # turn this into a dataset
        data_set = DataSet(data=data, prop_keys=prop_keys)
        data_set.random_split(
            n_train=n_train,
            n_valid=n_valid,
            n_test=0,
            r_cut=r_cut,
            training=True,
            mic=mic,
            seed=seed_data,
        )

        if shift_mean:
            data_set.shift_x_by_mean_x(x=pn.energy)

        data_set.save_splits_to_file(ckpt_dir.absolute(), file_splits)
        data_set.save_scales(ckpt_dir.absolute(), file_scales)

        d = data_set.get_data_split()
        scales = None
        if loss_variance_scaling:
            scales = get_scales_with_variance_scaling(d, targets)

        data_tuple = DataTuple(inputs=inputs, targets=targets, prop_keys=prop_keys)
        train_ds = data_tuple(d["train"])
        valid_ds = data_tuple(d["valid"])

        return data_set, n_train, scales, train_ds, valid_ds

    ckpt_dir.mkdir(exist_ok=overwrite_module)
    data_set, n_train, scales, train_ds, valid_ds = get_dataset_scales_splits(
        data,
        stress=use_stress,
        shift_mean=shift_mean,
        loss_variance_scaling=loss_variance_scaling,
        ckpt_dir=ckpt_dir,
    )

    # loss weights
    def get_loss_weights(
        weight_energy: float, weight_forces: float, weight_stress: float
    ) -> dict:
        """Gett effective loss weights normalized to 1"""

        _loss_weights = {pn.energy: weight_energy, pn.force: weight_forces}
        if weight_stress is not None:  # <- we want to train with stress
            _loss_weights.update({pn.stress: weight_stress})
        total_loss_weight = sum(x for x in _loss_weights.values())
        loss_weights = {k: v / total_loss_weight for k, v in _loss_weights.items()}

        return loss_weights

    loss_weights = get_loss_weights(
        weight_energy=we, weight_forces=wf, weight_stress=ws
    )

    def get_net(prop_keys, F, L, l_min, l_max, mic, r_cut):
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

    net = get_net(prop_keys, F, L, l_min, l_max, mic, r_cut)

    def get_obs_fn(targets, net):
        if pn.force in targets:
            if pn.stress in targets:
                obs_fn = get_energy_force_stress_fn(net)
            else:
                obs_fn = get_obs_and_force_fn(net)
        else:
            obs_fn = get_observable_fn(net)

        obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))
        return obs_fn

    obs_fn = get_obs_fn(targets, net)

    opt = Optimizer(clip_by_global_norm=clip_by_global_norm)

    # batch sizes
    def get_batch_sizes(
        n_train, size_batch=None, size_batch_training=None, size_batch_validation=None
    ):
        if size_batch is None:
            size_batch = autoset_batch_size(n_train)
        if size_batch_training is None:
            size_batch_training = size_batch
        if size_batch_validation is None:
            size_batch_validation = size_batch

        return size_batch, size_batch_training, size_batch_validation

    size_batch, size_batch_training, size_batch_validation = get_batch_sizes(
        n_train, size_batch, size_batch_training, size_batch_validation
    )

    coach = Coach(
        inputs=inputs,
        targets=targets,
        epochs=epochs,
        training_batch_size=size_batch_training,
        validation_batch_size=size_batch_validation,
        loss_weights=loss_weights,
        ckpt_dir=ckpt_dir.as_posix(),
        data_path=file_data.as_posix(),
        net_seed=seed_model,
        training_seed=seed_training,
        stop_lr_min=lr_stop,
    )

    loss_fn = get_loss_fn(
        obs_fn=obs_fn,
        weights=loss_weights,
        scales=scales,
        prop_keys=prop_keys,
    )

    def _create_train_state(
        net,
        opt,
        coach,
        train_ds,
        lr,
        lr_decay_exp_transition_steps,
        lr_decay_exp_decay_factor,
    ):

        lr_decay_exp = {
            "exponential": {
                "transition_steps": lr_decay_exp_transition_steps,
                "decay_factor": lr_decay_exp_decay_factor,
            }
        }

        tx = opt.get(learning_rate=lr)
        inputs = jax.tree_map(lambda x: jnp.array(x[0, ...]), train_ds[0])
        params = net.init(jax.random.PRNGKey(coach.net_seed), inputs)
        return create_train_state(
            net,
            params,
            tx,
            polyak_step_size=None,
            plateau_lr_decay=None,
            scheduled_lr_decay=lr_decay_exp,
            lr_warmup=None,
        )

    train_state, h_train_state = _create_train_state(
        net,
        opt,
        coach,
        train_ds,
        lr,
        lr_decay_exp_transition_steps,
        lr_decay_exp_decay_factor,
    )

    def bundle_and_save_dicts(
        net, opt, coach, data_set, ckpt_dir="module", filename="hyperparameters.json"
    ):

        h_net = net.__dict_repr__()
        h_opt = opt.__dict_repr__()
        h_coach = coach.__dict_repr__()
        h_dataset = data_set.__dict_repr__()
        h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])
        save_dict(path=ckpt_dir, filename=filename, data=h, exists_ok=True)

        return h

    h = bundle_and_save_dicts(net, opt, coach, data_set, ckpt_dir=ckpt_dir)

    # wandb
    def initialize_wandb(wandb_name, wandb_group, wandb_project, config):
        use_wandb = False
        if not all(x is None for x in (wandb_name, wandb_group, wandb_project)):
            kw = {"group": wandb_group, "project": wandb_project, "name": wandb_name}
            echo("... initialize WanDB with ")
            echo(kw)
            wandb.init(config=config, **kw)
            use_wandb = True

        return use_wandb

    use_wandb = initialize_wandb(wandb_name, wandb_group, wandb_project, config=h)

    # Save parameters
    echo(f"... write input arguments to {outfile_inputs}")
    with open(outfile_inputs, "w") as f:
        json.dump(ctx.params, f, indent=1)

    echo("... go 🚀")

    asdf

    coach.run(
        train_state=train_state,
        train_ds=train_ds,
        valid_ds=valid_ds,
        loss_fn=loss_fn,
        ckpt_overwrite=True,
        eval_every_t=eval_every_t,
        log_every_t=1,
        restart_by_nan=True,
        use_wandb=use_wandb,
    )


if __name__ == "__main__":
    app()
