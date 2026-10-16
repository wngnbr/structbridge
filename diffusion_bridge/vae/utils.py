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
from typing import List, Tuple, Union
import time
import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from ..base import BaseModule
from .volume_decoders import generate_dense_grid_points

class FourierEmbedder(nn.Module):
    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class AutoEncoder(BaseModule):
    def configure(self) -> None:
        super().configure()

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def encode_kl_embed(self, latents: torch.FloatTensor, sample_posterior: bool = True):
        """Map hidden tokens [B, T, W] to posterior embeddings [B, T, C]."""
        posterior = None
        if self.cfg.embed_dim > 0:
            moments = self.pre_kl(latents)  # [B, T, 2C]: mean and log variance
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents
        return kl_embed, posterior


    def query(self, queries: torch.FloatTensor, latents: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


    @torch.no_grad()
    def extract_geometry(self,
                         latents: torch.FloatTensor,
                         extract_mesh_func: str = "mc",
                         bounds: Union[Tuple[float], List[float], float] = (-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
                         octree_depth: int = 8,
                         num_chunks: int = 100000,
                         use_flashVDM: bool = True,
                         ):

        batch_size = latents.shape[0]
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        if use_flashVDM:
            from .volume_decoders import HierarchicalVolumeDecoding
            volume_decoder = HierarchicalVolumeDecoding()
            grid_logits = volume_decoder(latents, self.query, \
                **{'bounds': bounds, 'octree_resolution': 2**octree_depth, 'num_chunks': num_chunks})
            grid_logits = grid_logits.cpu().float().numpy()
            grid_size = grid_logits.shape[1:4]

        else:
            start_time_generate_dense_grid_points = time.time()
            xyz_samples, grid_size, length = generate_dense_grid_points(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                octree_resolution=2**octree_depth,
                indexing="ij"
            )
            xyz_samples = torch.FloatTensor(xyz_samples.reshape(-1, 3))
            print(f"generate_dense_grid_points time: {time.time()-start_time_generate_dense_grid_points}")

            start_time_query_sdf = time.time()
            batch_logits = []
            for start in range(0, xyz_samples.shape[0], num_chunks):
                queries = xyz_samples[start: start + num_chunks, :].to(latents)
                batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

                logits = self.query(batch_queries, latents)
                batch_logits.append(logits.cpu())
            print(f"query_sdf time: {time.time()-start_time_query_sdf}")

            grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2])).float().numpy()

        start_time_extract_mesh = time.time()
        mesh_v_f = []
        has_surface = np.zeros((batch_size,), dtype=np.bool_)
        for i in range(batch_size):
            try:
                if extract_mesh_func == "mc":
                    from skimage import measure
                    vertices, faces, normals, _ = measure.marching_cubes(grid_logits[i], 0)
                    vertices = vertices / grid_size * bbox_size + bbox_min
                    faces = faces[:, [2, 1, 0]]
                elif extract_mesh_func == "diffmc":
                    from diso import DiffMC
                    diffmc = DiffMC(dtype=torch.float32).to(latents.device)
                    vertices, faces = diffmc(-torch.tensor(grid_logits[i]).float().to(latents.device), isovalue=0)
                    vertices = vertices * 2 - 1
                    vertices = vertices.cpu().numpy()
                    faces = faces.cpu().numpy()
                    faces = faces[:, [2, 1, 0]]
                elif extract_mesh_func == "diffdmc":
                    from diso import DiffDMC
                    diffmc = DiffDMC(dtype=torch.float32).to(latents.device)
                    vertices, faces = diffmc(-torch.tensor(grid_logits[i]).float().to(latents.device), isovalue=0)
                    vertices = vertices * 2 - 1
                    vertices = vertices.cpu().numpy()
                    faces = faces.cpu().numpy()
                    faces = faces[:, [2, 1, 0]]
                else:
                    raise NotImplementedError(f"{extract_mesh_func} not implement")
                mesh_v_f.append((vertices.astype(np.float32), np.ascontiguousarray(faces.astype(np.int64))))
                has_surface[i] = True
            except:
                mesh_v_f.append((None, None))
                has_surface[i] = False

        print(f"extract_mesh time: {time.time()-start_time_extract_mesh}")
        return mesh_v_f, has_surface


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: Union[torch.Tensor, List[torch.Tensor]], deterministic=False, feat_dim=1):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                        + self.var - 1.0 - self.logvar,
                                        dim=dims)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=dims)

    def nll(self, sample, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
