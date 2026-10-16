"""Bridge training with frozen shape encoding and complete training checkpoints."""

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import RAdam

from . import logger
from .nn import update_ema
from .pc_matching import emd_matching
from .resample import RealUniformSampler
from ..checkpoints import (
    TRAINING_FORMAT, TRAINING_VERSION, atomic_save_checkpoint,
    capture_rng_state, restore_rng_state, validate_resume_config,
)
from ..config import to_dict


class TrainLoop:
    def __init__(self, model, diffusion, vae, train_data, config, workdir, resume=None):
        self.model = model
        self.diffusion = diffusion
        self.vae = vae
        self.data = train_data
        self.config = config
        self.workdir = Path(workdir)
        self.device = next(model.parameters()).device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.settings = config.training
        self.model.train()
        self.ddp_model = model
        if self.world_size > 1:
            device_ids = [self.device.index] if self.device.type == "cuda" else None
            self.ddp_model = DDP(model, device_ids=device_ids)
        self.opt = RAdam(model.parameters(), lr=self.settings.lr, weight_decay=self.settings.weight_decay)
        if self.settings.use_fp16 and self.device.type != "cuda":
            raise ValueError("FP16 training requires CUDA; set training.use_fp16=false for CPU")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.settings.use_fp16)
        self.schedule_sampler = RealUniformSampler(diffusion)
        # Initialize after DDP has synchronized model parameters across ranks.
        self.ema = {str(rate): [parameter.detach().clone() for parameter in model.parameters()]
                    for rate in self.settings.ema_rates}
        self.step = 0
        self.samples_seen = 0
        self.last_saved_step = None
        if resume is not None:
            self.restore(resume)

    def restore(self, checkpoint):
        if checkpoint.get("format") != TRAINING_FORMAT:
            raise ValueError("Resume requires a complete training checkpoint; use --pretrained_ckpt for weights only")
        if checkpoint["world_size"] != self.world_size:
            raise ValueError("Exact training resume requires the original world size")
        validate_resume_config(checkpoint["config"], to_dict(self.config))
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.opt.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        parameter_names = [name for name, _ in self.model.named_parameters()]
        if set(checkpoint["ema"]) != set(self.ema):
            raise ValueError("EMA rates differ from the saved training state")
        for rate in self.ema:
            state = checkpoint["ema"][rate]
            if set(state) != set(parameter_names):
                raise ValueError("EMA parameters differ from the model")
            self.ema[rate] = [state[name].to(self.device).clone() for name in parameter_names]
        self.step = checkpoint["step"]
        self.samples_seen = checkpoint["samples_seen"]
        rank_state = checkpoint["rank_states"][self.rank]
        self.data.batch_sampler.load_state_dict(rank_state["sampler"])
        # Restore last: construction, model loading and optimizer initialization
        # must not disturb the stochastic sequence of the resumed next step.
        restore_rng_state(rank_state["rng"])
        self.last_saved_step = self.step
        logger.log(f"Restored complete training state at step {self.step}")

    def run_loop(self):
        if self.settings.max_steps and self.step >= self.settings.max_steps:
            logger.log(f"Checkpoint already reached max_steps={self.settings.max_steps}")
            return
        iterator = iter(self.data)
        while not self.settings.max_steps or self.step < self.settings.max_steps:
            target_batch, condition_batch = next(iterator)
            target_latents, condition_latents = emd_matching(
                self.vae.encode(target_batch), self.vae.encode(condition_batch)
            )
            updated = self.run_step(target_latents, {"xT": condition_latents})
            self.data.batch_sampler.advance()
            self.samples_seen += target_latents.shape[0] * self.world_size
            if not updated:
                logger.log("AMP skipped an optimizer update; step, EMA and LR were not advanced")
                continue
            logger.logkv("step", self.step)
            logger.logkv("samples", self.samples_seen)
            if self.step % self.settings.log_interval == 0:
                values = logger.dumpkvs()
                if self.rank == 0 and self.settings.tracking_mode != "disabled":
                    import wandb
                    wandb.log(values, step=self.step)
            if self.settings.save_interval and self.step % self.settings.save_interval == 0:
                self.save()
        if self.settings.save_final and self.last_saved_step != self.step:
            self.save()
        logger.log(f"Completed {self.step} optimizer updates.")

    def run_step(self, target_latents, condition):
        """Accumulate microbatches and return whether the optimizer updated."""
        self.opt.zero_grad(set_to_none=True)
        batch_size = target_latents.shape[0]
        microbatch = self.settings.microbatch
        microbatch = batch_size if microbatch < 0 else min(microbatch, batch_size)
        for start in range(0, batch_size, microbatch):
            stop = min(start + microbatch, batch_size)
            last = stop == batch_size
            sync = self.ddp_model.no_sync() if isinstance(self.ddp_model, DDP) and not last else nullcontext()
            # no_sync must cover both forward and backward.
            with sync:
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.settings.use_fp16):
                    target = target_latents[start:stop].to(self.device)
                    model_kwargs = {key: value[start:stop].to(self.device) for key, value in condition.items()}
                    timesteps, weights = self.schedule_sampler.sample(len(target), self.device)
                    losses = self.diffusion.training_bridge_losses(self.ddp_model, target, timesteps,
                                                                    model_kwargs=model_kwargs)
                    loss = (losses["loss"] * weights).mean() * len(target) / batch_size
                self.scaler.scale(loss).backward()
                logger.logkv_mean("loss", losses["loss"].mean().item())
                logger.logkv_mean("latent_mse", losses["xs_mse"].mean().item())
        self.scaler.unscale_(self.opt)
        gradient_norm = torch.linalg.vector_norm(torch.stack([
            parameter.grad.detach().float().norm() for parameter in self.model.parameters()
            if parameter.grad is not None
        ])).item()
        logger.logkv_mean("grad_norm", gradient_norm)
        old_scale = self.scaler.get_scale()
        if not self.settings.use_fp16 and not torch.isfinite(torch.tensor(gradient_norm)):
            raise FloatingPointError("Nonfinite gradients in FP32 training")
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.scaler.get_scale() < old_scale:
            return False
        self.step += 1
        for rate, parameters in self.ema.items():
            update_ema(parameters, self.model.parameters(), rate=float(rate))
        if self.settings.lr_anneal_steps:
            fraction = min(self.step / self.settings.lr_anneal_steps, 1.0)
            lr = self.settings.lr * (1 - fraction) + 1e-7 * fraction
            for group in self.opt.param_groups:
                group["lr"] = lr
        return True

    def save(self):
        local_state = {"rng": capture_rng_state(), "sampler": self.data.batch_sampler.state_dict()}
        rank_states = [None] * self.world_size
        if self.world_size > 1:
            dist.all_gather_object(rank_states, local_state)
        else:
            rank_states[0] = local_state
        if self.rank == 0:
            names = [name for name, _ in self.model.named_parameters()]
            checkpoint = {
                "format": TRAINING_FORMAT, "version": TRAINING_VERSION,
                "model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                "ema": {rate: dict(zip(names, values)) for rate, values in self.ema.items()},
                "scaler": self.scaler.state_dict(), "step": self.step,
                "samples_seen": self.samples_seen, "config": to_dict(self.config),
                "rank_states": rank_states, "world_size": self.world_size,
            }
            path = self.workdir / f"checkpoint_{self.step:06d}.pt"
            atomic_save_checkpoint(checkpoint, path)
            logger.log(f"Saved complete training state: {path}")
        if self.world_size > 1:
            dist.barrier()
        self.last_saved_step = self.step
