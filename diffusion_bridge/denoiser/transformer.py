from dataclasses import dataclass

import torch
import torch.nn as nn
import math

from ..base import BaseModule
from .blocks import DiTBlock, TimestepEmbedder, AdaptiveOutputLayer, auto_grad_checkpoint

class BridgeDenoiser(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        input_channels: int
        output_channels: int
        width: int
        layers: int
        heads: int
        init_scale: float
        drop_path: float
        qkv_bias: bool
        qkv_fuse: bool
        use_rms_norm: bool

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # timestep embedding
        self.time_embed = TimestepEmbedder(self.cfg.width)

        # x embedding
        self.x_embed = nn.Linear(self.cfg.input_channels, self.cfg.width, bias=True)

        # concept embedding
        self.concept_embed = nn.Linear(self.cfg.input_channels, self.cfg.width, bias=True)

        init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)
        drop_path = [x.item() for x in torch.linspace(0, self.cfg.drop_path, self.cfg.layers)]
        self.blocks = nn.ModuleList([
            DiTBlock(
                    width=self.cfg.width,
                    heads=self.cfg.heads,
                    init_scale=init_scale,
                    qkv_bias=self.cfg.qkv_bias,
                    qkv_fuse=self.cfg.qkv_fuse,
                    use_rms_norm=self.cfg.use_rms_norm,
                    use_flash=True,
                    drop_path=drop_path[i]
            )
            for i in range(self.cfg.layers)
        ])

        self.t_block = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.cfg.width, 6 * self.cfg.width, bias=True)
                    )

        # Project latent tokens back to the VAE embedding dimension.
        self.final_layer = AdaptiveOutputLayer(self.cfg.width, self.cfg.output_channels, self.cfg.use_rms_norm)

        self.identity_initialize()

    def identity_initialize(self):
        for block in self.blocks:
            nn.init.constant_(block.attn.c_proj.weight, 0)
            nn.init.constant_(block.attn.c_proj.bias, 0)
            nn.init.constant_(block.cross_attn.c_proj.weight, 0)
            nn.init.constant_(block.cross_attn.c_proj.bias, 0)
            nn.init.constant_(block.mlp.c_proj.weight, 0)
            nn.init.constant_(block.mlp.c_proj.bias, 0)

    def forward(self,
                model_input: torch.FloatTensor,
                timestep: torch.FloatTensor,
                context: torch.FloatTensor):

        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.FloatTensor): [bs,]
            context (torch.FloatTensor): [bs, context_tokens, c]

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]

        """

        t_emb = self.time_embed(timestep)

        cond_emb = self.concept_embed(context)

        latent = self.x_embed(model_input)

        modulation = self.t_block(t_emb).unsqueeze(dim=1)
        for block in self.blocks:
            latent = auto_grad_checkpoint(block, latent, cond_emb, modulation)

        latent = self.final_layer(latent, t_emb)

        return latent
