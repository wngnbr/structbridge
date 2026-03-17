from dataclasses import dataclass, field
import numpy as np
import torch
from skimage import measure
from einops import repeat, rearrange

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.ops import generate_dense_grid_points
from craftsman.utils.typing import *
from craftsman.utils.misc import get_rank
import torch.distributed as dist
import os
from craftsman.models.geometry.utils import Mesh
import trimesh
from torchvision.utils import save_image,make_grid
from craftsman.systems.utils import *
import time


@craftsman.register("shape-autoencoder-system")
class ShapeAutoEncoderSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)
        sample_posterior: bool = True

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = craftsman.find(self.cfg.shape_model_type)(self.cfg.shape_model)


    def encoder_wng(self, batch: Dict[str, Any], split: str):
        
        kl_embed = self.shape_model.encoder_wng(
            batch["coarse_surface"].cuda(),  # xyz + normal
            batch["sharp_surface"].cuda(),  # xyz + normal
            sample_posterior=self.cfg.sample_posterior,
            split= split
        )

        return kl_embed

    def decoder_wng(self, kl_embed, batch):
        
        # latents, logits, mean_value, variance_value = self.shape_model.decoder_wng(kl_embed, batch["rand_points"].cuda())
        latents, logits, mean_value, variance_value = self.shape_model.decoder_wng(kl_embed, kl_embed)

        return latents

        # mesh_v_f, has_surface = self.shape_model.extract_geometry_by_diffdmc(latents, octree_depth=9)
        mesh_v_f, has_surface = self.shape_model.extract_geometry(latents, octree_depth=9)

        assert latents.shape[0] == len(mesh_v_f)
        batch_size = latents.shape[0]

        for i in range(batch_size):

            file_path = f"it{self.true_global_step}/{os.path.basename(batch['uid'][i])}".replace(".npz",".obj")
            if not os.path.exists(file_path):
                self.save_mesh(
                    file_path,
                    mesh_v_f[i][0], mesh_v_f[i][1]
            )


    def forward(self, batch: Dict[str, Any],split: str) -> Dict[str, Any]:
        num = batch["number_sharp"]
        rand_points = batch["rand_points"] 

        target = batch["sdf"]
        coarse_target = target[:,num:]
        sharp_target = target[:,:num]
        criteria = torch.nn.MSELoss()
        
        _, latents, posterior, logits, mean_value, variance_value = self.shape_model(
            batch["coarse_surface"],  # xyz + normal
            batch["sharp_surface"],  # xyz + normal
            rand_points, 
            sample_posterior=self.cfg.sample_posterior,
            split= split
        )

        loss_kl = posterior.kl()
        loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
        coarse_logits = logits[:,num:]
        sharp_logits = logits[:,:num]
        return {
            "loss_coarse_logits": criteria(coarse_logits, coarse_target).mean(),
            "loss_sharp_logits": criteria(sharp_logits, sharp_target).mean(),
            "loss_kl": loss_kl,
            "overall_logits": logits,
            "coarse_logits": coarse_logits,
            "sharp_logits": sharp_logits,
            "overall_target": target,
            "coarse_target": coarse_target,
            "sharp_target": sharp_target,
            "latents": latents,
            "mean_value": mean_value,
            "variance_value": variance_value
        }

    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()
        device = batch['coarse_surface'].device
        out = self(batch,'val')
        try:
            mesh_v_f, has_surface = self.shape_model.extract_geometry_by_diffdmc(out["latents"],octree_depth=9)
            file_path = f"it{self.true_global_step}/{os.path.basename(batch['uid'][0])}".replace(".npz",".obj")
            if not os.path.exists(file_path):
                self.save_mesh(
                    file_path,
                    mesh_v_f[0][0], mesh_v_f[0][1]
            )
        except Exception as e:
            print(f"ERROR: in processing batch: {batch}. Error: {e}")
            return


    def on_validation_epoch_end(self):
        pass