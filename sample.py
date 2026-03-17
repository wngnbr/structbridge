"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch
import torchvision.utils as vutils
import torch.distributed as dist

from ddbm import dist_util, logger
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from ddbm.karras_diffusion import karras_sample

from datasets import load_data

from pathlib import Path

import craftsman_models

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.config import ExperimentConfig, load_config

from datetime import datetime

import trimesh


def main():
    args = create_argparser().parse_args()
    args.use_fp16 = False

    cate = "all"

    workdir = os.path.join("sample", os.path.basename(args.model_path)[:-3])

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.model_path.replace("_adapted", "").split("_")
    step = int(split[-1].split(".")[0])
    time_str = datetime.now().strftime("%m-%d-%H-%M-%S")
    if args.sampler == "dbim":
        sample_dir = Path(workdir) / f"cate={cate}/split={args.split}/{cate}_{args.split}_{time_str}"
    elif args.sampler == "dbim_high_order":
        sample_dir = Path(workdir) / f"sample_{step}/split={args.split}/dbim_order={args.order}/steps={args.steps}"
    else:
        sample_dir = Path(workdir) / f"cate={cate}/split={args.split}/sample_{step}/time={time_str}"
    dist_util.setup_dist()
    if dist.get_rank() == 0:

        sample_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=str(sample_dir))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(dist_util.dev())

    if args.use_fp16:
        model = model.half()
    model.eval()

    logger.log("sampling...")

    all_images = []
    all_labels = []

    args.dataset = "dora"
    all_dataloaders = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=cate,
        include_test=False,
        seed=args.seed,
        num_workers=args.num_workers
    )
    if args.split == "train":
        dataloader = all_dataloaders[0]
    elif args.split == "test":
        dataloader = all_dataloaders[1]
    else:
        raise NotImplementedError
    args.num_samples = len(dataloader.dataset)

    # Dora
    config_path = "./craftsman/configs/shape-autoencoder/Dora-VAE-test.yaml"
    cfg: ExperimentConfig
    cfg = load_config(config_path, cli_args=[], n_gpus=1)
    cfg.trial_dir = sample_dir

    system: BaseSystem = craftsman.find('shape-autoencoder-system')(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    dora_vae = system.eval()
    for param in dora_vae.parameters():
        param.requires_grad = False
    dora_vae.to(dist_util.dev())


    num = 0
    for i, data in enumerate(dataloader):
        x0 = data[0]
        x0 = dora_vae.encoder_wng(x0, 'val')[0]

        y0 = data[1]
        y0 = dora_vae.encoder_wng(y0, 'val')[0]

        # # wng
        # y0 = x0

        model_kwargs = {"xT": y0}

        if "inpaint" in args.dataset:
            _, mask, label = data[2]
            mask = mask.to(dist_util.dev())
            label = label.to(dist_util.dev())
            model_kwargs["y"] = label
        else:
            mask = None

        indexes = np.array([i] * len(x0))
        sample, path, nfe, pred_x0, sigmas, _ = karras_sample(
            diffusion,
            model,
            y0,
            x0,
            steps=args.steps,
            mask=mask,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            churn_step_ratio=args.churn_step_ratio,
            eta=args.eta,
            order=args.order,
            seed=indexes + args.seed,
        )
        # sample = x0
        print("wng", sample.shape)

        outputs = dora_vae.decoder_wng(sample, data[1])

        for j in range(len(outputs)):
            mesh_v_f, has_surface = dora_vae.shape_model.extract_geometry(
                outputs[j:j+1], 
                octree_depth=9,
                extract_mesh_func="mc",
                use_flashVDM = True
            )
            print(f"Generating mesh {data[0]['uid'][j].split('.')[0]}")
            cur_mesh = trimesh.Trimesh(vertices=mesh_v_f[0][0], faces=mesh_v_f[0][1])
            cur_mesh.export(os.path.join(sample_dir, data[0]['uid'][j].replace('.npz', '.glb')))

        break
        

    logger.log(f"created {len(all_images) * args.batch_size * dist.get_world_size()} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if "inpaint" in args.dataset:
        labels = np.concatenate(all_labels, axis=0)
        labels = labels[: args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir, f"samples_{shape_str}_nfe{nfe}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
        if "inpaint" in args.dataset:
            shape_str = "x".join([str(x) for x in labels.shape])
            out_path = os.path.join(sample_dir, f"labels_{shape_str}_nfe{nfe}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, labels)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",  ## only used in bridge
        dataset="edges2handbags",
        clip_denoised=False,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        split="train",
        churn_step_ratio=0.0,
        rho=7.0,
        steps=40,
        model_path="",
        exp="",
        seed=42,
        num_workers=8,
        eta=1.0,
        order=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
