"""Train the concept-to-real diffusion bridge with a frozen shape VAE."""

import argparse
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from datasets import load_training_data
from diffusion_bridge.bridge import dist_util, logger
from diffusion_bridge.bridge.script_util import create_model_and_diffusion
from diffusion_bridge.bridge.train_util import TrainLoop
from diffusion_bridge.checkpoints import (
    TRAINING_FORMAT, bridge_weights, load_bridge_checkpoint, resolve_resume_path,
    validate_resume_config,
)
from diffusion_bridge.config import add_config_arguments, apply_cli, load_config, save_config, str2bool, to_dict
from diffusion_bridge.vae.model import load_vae


CLI_MAP = {
    "data_dir": "data.root", "category": "data.category", "category_filter": "data.category_filter",
    "num_workers": "data.num_workers", "vae_path": "vae.checkpoint", "workdir": "training.workdir",
    "exp": "training.experiment", "seed": "training.seed", "lr": "training.lr",
    "weight_decay": "training.weight_decay", "lr_anneal_steps": "training.lr_anneal_steps",
    "global_batch_size": "training.global_batch_size", "batch_size": "training.batch_size",
    "microbatch": "training.microbatch", "use_fp16": "training.use_fp16",
    "log_interval": "training.log_interval", "save_interval": "training.save_interval",
    "max_steps": "training.max_steps", "save_final": "training.save_final",
    "tracking_mode": "training.tracking_mode",
}


def main(args):
    # Discover checkpoint paths before validating the fully merged architecture.
    initial = apply_cli(load_config(args.config, validate=False), args, CLI_MAP, validate=False)
    resume_path = resolve_resume_path(args.resume_checkpoint, initial.training.workdir)
    source_path = resume_path or args.pretrained_ckpt
    checkpoint = load_bridge_checkpoint(source_path) if source_path else None
    embedded = checkpoint.get("config") if checkpoint and checkpoint.get("format") == TRAINING_FORMAT else None
    inherited = embedded if resume_path else ({key: embedded[key] for key in ("model", "vae", "diffusion")} if embedded else None)
    config = apply_cli(load_config(args.config, checkpoint_config=inherited, validate=False), args, CLI_MAP)
    if resume_path:
        if embedded is None:
            raise ValueError("This file contains only weights. Use --pretrained_ckpt to initialize a new run")
        validate_resume_config(embedded, to_dict(config))
    if not config.data.root:
        raise ValueError("Set --data_dir or data.root in the experiment configuration")
    dist_util.setup_dist()
    tracking = False
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        seed = config.training.seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        batch_size = config.training.batch_size
        if batch_size == -1:
            if config.training.global_batch_size % world_size:
                raise ValueError("global_batch_size must be divisible by the process count")
            batch_size = config.training.global_batch_size // world_size
        if batch_size <= 0:
            raise ValueError("Each rank must have a positive batch size")
        workdir = config.training.workdir
        if not workdir and resume_path:
            workdir = str(resume_path.parent)
        if not workdir and rank == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            workdir = str(Path("workdir") / f"{config.training.experiment}-{timestamp}")
        shared = [workdir]
        dist.broadcast_object_list(shared, src=0)
        workdir = Path(shared[0])
        config.training.workdir = str(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        logger.configure(dir=str(workdir))
        if rank == 0:
            # Do not overwrite the original run's config when resuming.
            name = f"config_resume_{checkpoint['step']:06d}.yaml" if resume_path else "config.yaml"
            save_config(config, workdir / name)
            if config.training.tracking_mode != "disabled":
                import wandb
                wandb.init(project="diffusion_bridge", name=config.training.experiment,
                           config=to_dict(config), mode=config.training.tracking_mode)
                tracking = True
        model, diffusion = create_model_and_diffusion(config)
        if checkpoint is not None:
            model.load_state_dict(bridge_weights(checkpoint), strict=True)
        model.to(dist_util.dev())
        vae = load_vae(config=config.vae, device=dist_util.dev())
        data = load_training_data(config, batch_size, world_size, rank)
        loop = TrainLoop(model, diffusion, vae, data, config, workdir,
                         resume=checkpoint if resume_path else None)
        del checkpoint
        loop.run_loop()
    finally:
        if tracking:
            import wandb
            wandb.finish()
        dist.destroy_process_group()


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--resume_checkpoint", help="Complete training state, or auto with --workdir")
    sources.add_argument("--pretrained_ckpt", help="Initialize a new run from bridge weights")
    default = load_config()
    for name, key in CLI_MAP.items():
        section, field = key.split(".")
        value = default[section][field]
        value_type = str2bool if isinstance(value, bool) else type(value)
        parser.add_argument(f"--{name}", type=value_type, default=None, help=f"Override {key}")
    return parser


if __name__ == "__main__":
    main(create_argparser().parse_args())
