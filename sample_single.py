import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime
import trimesh

# 原有依赖保留
from ddbm import dist_util, logger
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from ddbm.karras_diffusion import karras_sample

import craftsman_models
import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.config import ExperimentConfig, load_config


def main():
    args = create_argparser().parse_args()
    args.use_fp16 = False

    # ========== 核心修改：指定单个npz文件路径 ==========
    if not os.path.exists(args.npz_path):
        raise FileNotFoundError(f"指定的npz文件不存在: {args.npz_path}")
    
    # 构建输出目录
    workdir = os.path.join("sample", os.path.basename(args.model_path)[:-3])
    split = args.model_path.replace("_adapted", "").split("_")
    step = int(split[-1].split(".")[0]) if len(split) > 1 and split[-1].endswith(".pt") else "custom"
    time_str = datetime.now().strftime("%m-%d-%H-%M-%S")
    
    # 输出目录：基于npz文件名创建，方便识别
    npz_basename = os.path.basename(args.npz_path).replace(".npz", "")
    if args.sampler == "dbim":
        sample_dir = Path(workdir) / f"single_sample/{npz_basename}_{time_str}"
    elif args.sampler == "dbim_high_order":
        sample_dir = Path(workdir) / f"single_sample/{npz_basename}_order={args.order}_steps={args.steps}"
    else:
        sample_dir = Path(workdir) / f"single_sample/{npz_basename}_{time_str}"
    
    # 初始化分布式环境（保留，兼容模型运行）
    dist_util.setup_dist()
    if dist.get_rank() == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=str(sample_dir))

    logger.log("creating model and diffusion...")
    # 创建模型和扩散器（原有逻辑保留）
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(dist_util.dev())
    if args.use_fp16:
        model = model.half()
    model.eval()

    logger.log("loading dora vae...")
    # 加载Dora VAE（原有逻辑保留）
    config_path = "./craftsman/configs/shape-autoencoder/Dora-VAE-test.yaml"
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

    logger.log("processing single npz file...")
    # ========== 核心修改：读取单个npz文件 ==========
    # 加载npz文件
    npz_data = np.load(args.npz_path, allow_pickle=True)
    data = [
        {
            'uid': np.array([os.path.basename(args.npz_path)]),  # 用于生成glb文件名
            'coarse_surface': torch.from_numpy(npz_data["fps_coarse_surface"][:,0,:].astype(np.float32)).unsqueeze(0).cuda(),
            'sharp_surface': torch.from_numpy(npz_data["fps_sharp_surface"][:,0,:].astype(np.float32)).unsqueeze(0).cuda(),
        }
    ]
    
    # 兼容原有代码的x0/y0提取逻辑
    x0 = dora_vae.encoder_wng(data[0], 'val')[0]
    y0 = dora_vae.encoder_wng(data[0], 'val')[0]

    # 模型参数构造（原有逻辑保留）
    model_kwargs = {"xT": y0}
    mask = None  # 单个文件无inpaint，mask设为None

    # 采样（原有逻辑保留，适配单个样本）
    indexes = np.array([0])  # 单个样本，索引设为0
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

    # 解码生成网格（原有逻辑保留）
    outputs = dora_vae.decoder_wng(sample, data[0])
    
    # 提取几何信息并保存为glb
    mesh_v_f, has_surface = dora_vae.shape_model.extract_geometry(
        outputs, 
        octree_depth=9,
        extract_mesh_func="mc",
        use_flashVDM=True
    )
    glb_filename = os.path.basename(args.npz_path).replace(".npz", ".glb")
    glb_save_path = os.path.join(sample_dir, glb_filename)
    cur_mesh = trimesh.Trimesh(vertices=mesh_v_f[0][0], faces=mesh_v_f[0][1])
    cur_mesh.export(glb_save_path)
    
    logger.log(f"Success! GLB file saved to: {glb_save_path}")
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        # ========== 新增参数：指定单个npz文件路径 ==========
        data_dir="",  # 保留但无需使用
        dataset="dora",  # 固定为dora
        clip_denoised=False,
        batch_size=1,  # 单个样本，batch_size设为1
        sampler="dbim",
        split="custom",  # 自定义，无train/test区分
        churn_step_ratio=0.0,
        rho=7.0,
        steps=30,
        model_path="./workdir/model_300000.pt",  # 模型权重路径（必填）
        exp="",
        seed=42,
        num_workers=0,  # 单个样本无需多线程
        eta=0.9,
        order=1,
        noise_schedule="i2sb_cond"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    # 突出显示新增的npz_path参数
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the single npz file to process")
    return parser


if __name__ == "__main__":
    main()