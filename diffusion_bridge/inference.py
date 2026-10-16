"""Shared NPZ-to-mesh inference for single files and directories."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import trimesh

from .bridge import dist_util, logger
from .bridge.karras_diffusion import karras_sample
from .bridge.script_util import create_model_and_diffusion
from .checkpoints import TRAINING_FORMAT, bridge_weights, load_bridge_checkpoint
from .config import add_config_arguments, apply_cli, load_config, save_config, str2bool
from .vae.model import load_vae


def load_surface(path):
    """Read the two point groups (xyz + normal) without loading supervision."""
    batch = {"uid": str(path)}
    with np.load(path, allow_pickle=False) as data:
        for name in ("coarse_surface", "sharp_surface"):
            points = data[f"fps_{name}"]
            if points.ndim != 3 or points.shape[1:] != (1, 6) or not len(points):
                raise ValueError(f"{path}: fps_{name} must have shape (N, 1, 6)")
            if not np.isfinite(points).all():
                raise ValueError(f"{path}: fps_{name} contains nonfinite values")
            batch[name] = torch.from_numpy(points[:, 0, :].astype(np.float32)).unsqueeze(0)
    return batch


class BridgePipeline:
    def __init__(self, config, checkpoint):
        self.config = config
        self.device = dist_util.dev()
        logger.log("creating model and diffusion...")
        self.model, self.diffusion = create_model_and_diffusion(config)
        self.model.load_state_dict(bridge_weights(checkpoint), strict=True)
        self.model.to(self.device).eval()
        if config.sampling.use_fp16 and self.device.type != "cuda":
            raise ValueError("FP16 sampling requires CUDA")
        logger.log("loading shape VAE...")
        self.vae = load_vae(config=config.vae, device=self.device)

    @torch.no_grad()
    def sample(self, path, output_dir, index=0):
        args = self.config.sampling
        batch = load_surface(path)
        # Preserve the original two posterior draws and FPS RNG ordering so
        # existing seeds produce exactly the same results after the refactor.
        reference_latents = self.vae.encode(batch)[0]
        condition = self.vae.encode(batch)[0]
        # Autocast only the bridge sampler; retain FP32 VAE encoding/geometry.
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=args.use_fp16):
            sample, _, nfe, _, _, _ = karras_sample(
                self.diffusion, self.model, condition, reference_latents,
                steps=args.steps, model_kwargs={"xT": condition}, device=self.device,
                clip_denoised=args.clip_denoised, sampler=args.sampler,
                churn_step_ratio=args.churn_step_ratio, eta=args.eta,
                order=args.order, rho=args.rho, seed=np.array([index + args.seed]),
            )
        decoded = self.vae.decode(sample)
        meshes, has_surface = self.vae.shape_model.extract_geometry(
            decoded, octree_depth=args.octree_depth, extract_mesh_func="mc", use_flashVDM=True,
        )
        if not has_surface[0]:
            raise RuntimeError(f"No surface could be extracted for {path}")
        output = Path(output_dir) / f"{Path(path).stem}.glb"
        mesh = trimesh.Trimesh(vertices=meshes[0][0], faces=meshes[0][1])
        mesh.export(output)
        logger.log(f"Saved {output} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces, NFE={nfe})")
        return output


CLI_MAP = {
    "model_path": "sampling.model_path", "vae_path": "vae.checkpoint",
    "output_dir": "sampling.output_dir", "use_fp16": "sampling.use_fp16",
    "clip_denoised": "sampling.clip_denoised", "sampler": "sampling.sampler",
    "steps": "sampling.steps", "churn_step_ratio": "sampling.churn_step_ratio",
    "rho": "sampling.rho", "seed": "sampling.seed", "eta": "sampling.eta",
    "order": "sampling.order", "octree_depth": "sampling.octree_depth",
}


def run_sampling(paths, args, batch=False):
    initial = apply_cli(load_config(args.config, validate=False), args, CLI_MAP, validate=False)
    checkpoint = load_bridge_checkpoint(initial.sampling.model_path)
    embedded = checkpoint.get("config") if checkpoint.get("format") == TRAINING_FORMAT else None
    config = apply_cli(load_config(args.config, checkpoint_config=embedded, validate=False), args, CLI_MAP)
    # The requested source remains the selected checkpoint, even when the saved
    # training config contains a different inference example path.
    config.sampling.model_path = initial.sampling.model_path
    settings = config.sampling
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if not paths:
        raise ValueError("No NPZ files found")
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    dist_util.setup_dist()
    try:
        timestamp = datetime.now().strftime("%m-%d-%H-%M-%S") if dist.get_rank() == 0 else None
        timestamps = [timestamp]
        dist.broadcast_object_list(timestamps, src=0)
        timestamp = timestamps[0]
        suffix = f"batch/{timestamp}" if batch else f"single_sample/{Path(paths[0]).stem}_{timestamp}"
        output_dir = Path(settings.output_dir) if settings.output_dir else Path("sample") / Path(settings.model_path).stem / suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.configure(dir=str(output_dir))
        if dist.get_rank() == 0:
            save_config(config, output_dir / "config.yaml")
            (output_dir / "inputs.json").write_text(json.dumps([str(path) for path in paths], indent=2) + "\n")
        pipeline = BridgePipeline(config, checkpoint)
        del checkpoint
        for index, path in enumerate(paths):
            if index % dist.get_world_size() == dist.get_rank():
                pipeline.sample(path, output_dir, index)
        dist.barrier()
        logger.log("sampling complete")
    finally:
        dist.destroy_process_group()


def create_argparser(batch=False, require_input=True):
    parser = argparse.ArgumentParser(description="Sample meshes with the shape diffusion bridge")
    add_config_arguments(parser)
    default = load_config()
    for name, key in CLI_MAP.items():
        section, field = key.split(".")
        value = default[section][field]
        value_type = str2bool if isinstance(value, bool) else type(value)
        parser.add_argument(f"--{name}", type=value_type, default=None, help=f"Override {key}")
    if batch:
        parser.add_argument("--data_dir", required=True, help="Directory of preprocessed NPZ files")
        parser.add_argument("--limit", type=int, default=0, help="Maximum number of files; 0 means all")
    else:
        parser.add_argument("--npz_path", required=require_input, help="Preprocessed shape NPZ file")
    return parser
