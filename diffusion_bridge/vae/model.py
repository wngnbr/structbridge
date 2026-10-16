"""Frozen shape VAE used by bridge training and mesh sampling."""

import torch
from torch import nn

from ..config import load_config
from ..checkpoints import load_vae_state_dict
from .autoencoder import ShapeAutoEncoder


class ShapeVAE(nn.Module):
    def __init__(self, config, sample_posterior=True):
        super().__init__()
        self.shape_model = ShapeAutoEncoder(config)
        self.sample_posterior = sample_posterior

    @torch.no_grad()
    def encode(self, batch):
        """Return embeddings [B, 2K, C] and two FPS coordinate groups [B, K, 3]."""
        device = next(self.parameters()).device
        _, latents, _, coarse_points, sharp_points = self.shape_model.encode(
            batch["coarse_surface"].to(device),
            batch["sharp_surface"].to(device),
            sample_posterior=self.sample_posterior,
        )
        return latents, coarse_points, sharp_points

    @torch.no_grad()
    def decode(self, latents):
        """Map VAE embeddings [B, 2K, C] to mesh-query features [B, 2K, W]."""
        return self.shape_model.decode(latents)


def load_vae(checkpoint_path=None, device="cpu", config=None):
    """Load the frozen VAE from the experiment's resolved vae section."""
    config = load_config().vae if config is None else config
    path = config.checkpoint if checkpoint_path is None else checkpoint_path
    if not path:
        raise ValueError("vae.checkpoint must identify pretrained VAE weights")
    model = ShapeVAE(config.model, sample_posterior=config.sample_posterior)
    model.shape_model.load_state_dict(load_vae_state_dict(path), strict=True)
    return model.eval().requires_grad_(False).to(device)
