"""Small configuration base shared by the VAE and bridge denoiser."""

from dataclasses import dataclass

import torch
from torch import nn
from omegaconf import OmegaConf


class BaseModule(nn.Module):
    @dataclass
    class Config:
        pass

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = OmegaConf.structured(self.Config(**(cfg or {})))
        self.configure()
        self.register_buffer("_dummy", torch.zeros(0), persistent=False)

    def configure(self):
        pass
