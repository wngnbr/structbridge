#  Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from einops import repeat

from ..layers.perceiver import Perceiver
from ..layers.attention import ResidualCrossAttentionBlock
from ..layers.checkpoint import checkpoint
from ..base import BaseModule
from typing import Optional

from .utils import AutoEncoder, FourierEmbedder
from torch_cluster import fps
import numpy as np
class PerceiverCrossAttentionEncoder(nn.Module):
    def __init__(self,
                 use_downsample: bool,
                 num_latents: int,
                 tokens_per_group: int,
                 embedder: FourierEmbedder,
                 point_feats: int,
                 embed_point_feats: bool,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_ln_post: bool = False,
                 use_flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.tokens_per_group = tokens_per_group
        self.use_downsample = use_downsample
        self.embed_point_feats = embed_point_feats

        if not self.use_downsample:
            self.query = nn.Parameter(torch.randn((num_latents, width)) * 0.02)

        self.embedder = embedder
        if self.embed_point_feats:
            self.coarse_input_proj = nn.Linear(self.embedder.out_dim * 2, width)
        else:
            self.coarse_input_proj = nn.Linear(self.embedder.out_dim + point_feats, width)
            self.sharp_input_proj = nn.Linear(self.embedder.out_dim + point_feats, width)

        self.coarse_cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=False
        )

        self.sharp_cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=False
        )

        self.self_attn = Perceiver(
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=use_checkpoint
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    def _forward(self, coarse_pc, sharp_pc, coarse_feats, sharp_feats):
        bs, N_coarse, D_coarse = coarse_pc.shape
        bs, N_sharp, D_sharp = sharp_pc.shape

        coarse_data = self.embedder(coarse_pc)
        if coarse_feats is not None:
            if self.embed_point_feats:
                coarse_feats = self.embedder(coarse_feats)
            coarse_data = torch.cat([coarse_data, coarse_feats], dim=-1)

        coarse_data = self.coarse_input_proj(coarse_data)

        sharp_data = self.embedder(sharp_pc)
        if sharp_feats is not None:
            if self.embed_point_feats:
                sharp_feats = self.embedder(sharp_feats)
            sharp_data = torch.cat([sharp_data, sharp_feats], dim=-1)
        sharp_data = self.sharp_input_proj(sharp_data)

        if self.use_downsample:
            # One weighted draw preserves the released model's NumPy RNG
            # ordering even though the frozen encoder uses a fixed token count.
            if min(N_coarse, N_sharp) < self.tokens_per_group:
                raise ValueError("Each point group must contain at least tokens_per_group points")
            coarse_ratios = np.array([float(self.tokens_per_group)]) / N_coarse
            ratio_coarse = np.random.choice(coarse_ratios, size=1, p=[1.0])[0]
            ratio_sharp = self.tokens_per_group / N_sharp

            flattened = coarse_pc.view(bs*N_coarse, D_coarse)
            batch = torch.arange(bs).to(coarse_pc.device)
            batch = torch.repeat_interleave(batch, N_coarse)
            pos = flattened
            idx = fps(pos, batch, ratio=ratio_coarse)
            query_coarse = coarse_data.view(bs*N_coarse, -1)[idx].view(bs, -1, coarse_data.shape[-1])

            # Keep the FPS positions aligned with the corresponding latent tokens.
            sliced_coarse_pc = coarse_pc.view(bs*N_coarse, -1)[idx].view(bs, -1, coarse_pc.shape[-1])

            flattened = sharp_pc.view(bs*N_sharp, D_sharp)
            batch = torch.arange(bs).to(sharp_pc.device)
            batch = torch.repeat_interleave(batch, N_sharp)
            pos = flattened
            idx = fps(pos, batch, ratio=ratio_sharp)
            query_sharp = sharp_data.view(bs*N_sharp, -1)[idx].view(bs, -1, sharp_data.shape[-1])

            # Keep the FPS positions aligned with the corresponding latent tokens.
            sliced_sharp_pc = sharp_pc.view(bs*N_sharp, -1)[idx].view(bs, -1, sharp_pc.shape[-1])

            query = torch.cat([query_coarse, query_sharp], dim=1)

        else:
            query = self.query
            query = repeat(query, "m c -> b m c", b=bs)


        latents_coarse = self.coarse_cross_attn(query, coarse_data)
        latents_sharp = self.sharp_cross_attn(query, sharp_data)
        latents = latents_coarse + latents_sharp

        latents = self.self_attn(latents)
        if self.ln_post is not None:
            latents = self.ln_post(latents)

        # Token order is coarse first, then sharp; matching relies on this order.
        return latents, sliced_coarse_pc, sliced_sharp_pc

    def forward(self, coarse_pc: torch.FloatTensor, sharp_pc: torch.FloatTensor, coarse_feats: Optional[torch.FloatTensor] = None, sharp_feats: Optional[torch.FloatTensor] = None):
        """Encode XYZ groups [B, N, 3] and normal features [B, N, 3].

        Returns hidden tokens [B, 2K, W] and the coarse/sharp FPS coordinates,
        each [B, K, 3]. K is tokens_per_group; W is the encoder width.
        """

        return self._forward(coarse_pc, sharp_pc, coarse_feats, sharp_feats)


class PerceiverCrossAttentionDecoder(nn.Module):

    def __init__(self,
                 num_latents: int,
                 out_dim: int,
                 embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.embedder = embedder

        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=use_checkpoint
        )

        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_dim)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)
        return logits


class ShapeAutoEncoder(AutoEncoder):
    r"""
    A VAE model for encoding shapes into latents and decoding latent representations into shapes.
    """

    @dataclass
    class Config(BaseModule.Config):
        use_downsample: bool
        tokens_per_group: int
        point_feats: int
        embed_point_feats: bool
        out_dim: int
        embed_dim: int
        embed_type: str
        num_freqs: int
        include_pi: bool
        width: int
        heads: int
        num_encoder_layers: int
        num_decoder_layers: int
        init_scale: float
        qkv_bias: bool
        use_ln_post: bool
        use_flash: bool
        use_checkpoint: bool

    cfg: Config

    def configure(self) -> None:
        super().configure()

        if self.cfg.embed_type != "fourier":
            raise ValueError("The shape VAE requires Fourier embeddings.")
        self.embedder = FourierEmbedder(num_freqs=self.cfg.num_freqs, include_pi=self.cfg.include_pi)
        init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)
        num_latents = 2 * self.cfg.tokens_per_group
        self.encoder = PerceiverCrossAttentionEncoder(
            use_downsample=self.cfg.use_downsample,
            tokens_per_group=self.cfg.tokens_per_group,
            embedder=self.embedder,
            num_latents=num_latents,
            point_feats=self.cfg.point_feats,
            embed_point_feats=self.cfg.embed_point_feats,
            width=self.cfg.width,
            heads=self.cfg.heads,
            layers=self.cfg.num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=self.cfg.qkv_bias,
            use_ln_post=self.cfg.use_ln_post,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )

        if self.cfg.embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(self.cfg.width, self.cfg.embed_dim * 2)
            self.post_kl = nn.Linear(self.cfg.embed_dim, self.cfg.width)
            self.latent_shape = (2 * self.cfg.tokens_per_group, self.cfg.embed_dim)
        else:
            self.latent_shape = (2 * self.cfg.tokens_per_group, self.cfg.width)

        self.transformer = Perceiver(
            n_ctx=num_latents,
            width=self.cfg.width,
            layers=self.cfg.num_decoder_layers,
            heads=self.cfg.heads,
            init_scale=init_scale,
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )

        # decoder
        self.decoder = PerceiverCrossAttentionDecoder(
            embedder=self.embedder,
            out_dim=self.cfg.out_dim,
            num_latents=num_latents,
            width=self.cfg.width,
            heads=self.cfg.heads,
            init_scale=init_scale,
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )


    def encode(self,
               coarse_surface: torch.FloatTensor,
               sharp_surface: torch.FloatTensor,
               sample_posterior: bool = True,
               ):
        """Encode coarse/sharp surfaces [B, N_coarse/sharp, 6].

        Returns hidden tokens [B, 2K, W], VAE embeddings [B, 2K, C],
        their posterior, and coarse/sharp FPS positions [B, K, 3].
        K is tokens_per_group; C is embed_dim; W is the model width.
        """

        coarse_pc, coarse_feats = coarse_surface[..., :3], coarse_surface[..., 3:]
        sharp_pc, sharp_feats = sharp_surface[..., :3], sharp_surface[..., 3:]
        shape_latents, sliced_coarse_pc, sliced_sharp_pc = self.encoder(coarse_pc, sharp_pc, coarse_feats, sharp_feats)
        kl_embed, posterior = self.encode_kl_embed(shape_latents, sample_posterior)

        # Return FPS coordinates alongside tokens for geometric matching.
        return shape_latents, kl_embed, posterior, sliced_coarse_pc, sliced_sharp_pc


    def decode(self,
               latents: torch.FloatTensor):
        """
        Args:
            latents (torch.FloatTensor): posterior embeddings [B, 2K, C]

        Returns:
            latents (torch.FloatTensor): decoded hidden features [B, 2K, W]
        """
        latents = self.post_kl(latents)  # [B, 2K, C] -> [B, 2K, W]

        return self.transformer(latents)


    def query(self,
              queries: torch.FloatTensor,
              latents: torch.FloatTensor):
        """
        Args:
            queries (torch.FloatTensor): [B, N, 3]
            latents (torch.FloatTensor): decoded hidden features [B, 2K, W]

        Returns:
            logits (torch.FloatTensor): [B, N, out_dim], implicit field values
        """
        logits = self.decoder(queries, latents)

        return logits
