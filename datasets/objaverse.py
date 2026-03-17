import math
import os
import json
from pathlib import Path

import random
import numpy as np 
import torch
from torch.utils.data import Dataset


class ObjaverseDataset_old(Dataset):

    def __init__(self, root: Path, cate: str, split: str) -> None:
        super().__init__()
        self.root = root
        self.cate = cate

        self.data_real = os.path.join(self.root, f'{cate}_real_glbs')
        self.data_concept = os.path.join(self.root, f'{cate}_concept_glbs')

        if split != 'train':
            self.data_real = os.path.join(self.root, f'{cate}_{split}_glbs')
            self.data_concept = os.path.join(self.root, f'{cate}_{split}_glbs')

        self.ids = []
        self.npz_list_real = []
        self.npz_list_concept = []
        for id in os.listdir(self.data_real):
            assert id.endswith(".npz")
            self.ids.append(id)
            self.npz_list_real.append(os.path.join(self.data_real, id))
            self.npz_list_concept.append(os.path.join(self.data_concept, id))

    def __len__(self) -> int:
        return len(self.ids)

    def _load_shape(self, path, index):
        data = np.load(path)
        coarse_fps_surface = data["fps_coarse_surface"]
        sharp_fps_surface = data['fps_sharp_surface']

        coarse_surface = coarse_fps_surface[:,0,:] #（m,1,p) >(m.p)

        nan_mask = np.isnan(coarse_surface)
        if np.any(nan_mask):
            print("nan exist in coarse surface")
        coarse_surface = np.where(nan_mask, 1, coarse_surface)

        sharp_surface = sharp_fps_surface[:,0,:]

        nan_mask = np.isnan(sharp_surface)
        if np.any(nan_mask):
            print("nan exist in sharp surface! ")
        sharp_surface = np.where(nan_mask, 1, sharp_surface)

        ret = {
            "uid": self.ids[index],
            "coarse_surface": coarse_surface.astype(np.float32),
            "sharp_surface": sharp_surface.astype(np.float32),
            "data":data
        }

        return ret

    def _load_shape_supervision(self, data):
        ret = {}
        coarse_rand_points = data['rand_points'][:,:3]
        coarse_sdfs = data['rand_points'][:,3]
        sharp_near_points = data['sharp_near_surface'][:,:3]
        sharp_sdfs = data['sharp_near_surface'][:,3]

        n_supervision = [21384, 10000, 10000]

        rng = np.random.default_rng()
        ind2 = rng.choice(sharp_near_points.shape[0], n_supervision[0], replace=False)
        ind3 = rng.choice(coarse_rand_points[:400000].shape[0], n_supervision[1], replace=False)
        ind4 = rng.choice(coarse_rand_points[400000:].shape[0], n_supervision[2], replace=False)
        rand_points2 = sharp_near_points[ind2] 
        rand_points3 = coarse_rand_points[:400000][ind3]
        rand_points4 = coarse_rand_points[400000:][ind4]
        rand_points = np.concatenate([rand_points2, rand_points3,rand_points4], axis=0) 
        ret["rand_points"] = rand_points.astype(np.float32)
        ret["number_sharp"] = n_supervision[0]

        sdf2 = sharp_sdfs[ind2]
        sdf3 = coarse_sdfs[:400000][ind3]
        sdf4 = coarse_sdfs[400000:][ind4]
        sdfs = np.concatenate([sdf2,sdf3,sdf4], axis=0)
        nan_mask = np.isnan(sdfs)
        if np.any(nan_mask):
            print("nan exist in sdfs")
        sdfs = np.where(nan_mask, 0, sdfs)
        ret["sdf"] = sdfs.flatten().astype(np.float32).clip(-0.015,0.015) / 0.015

        return ret

    def get_data(self, index):
        real_path = f'{self.npz_list_real[index]}'
        ret = self._load_shape(real_path, index)
        ret.update(self._load_shape_supervision(ret['data']))
        del ret['data']

        concept_path = f'{self.npz_list_concept[index]}'
        ret_c = self._load_shape(concept_path, index)
        ret_c.update(self._load_shape_supervision(ret_c['data']))
        del ret_c['data']

        return ret, ret_c
        
    def __getitem__(self, index):
        try:
            return self.get_data(index)
        except Exception as e:
            print(f"Error in {self.ids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))
        


class ObjaverseDataset(Dataset):

    def __init__(self, root: Path, cate: str, split: str) -> None:
        super().__init__()
        self.root = root
        self.cate = cate

        self.data_real = os.path.join(self.root, f'{cate}_real_glbs')
        self.data_concept = os.path.join(self.root, f'{cate}_concept_glbs')

        if split != 'train':
            self.data_real = os.path.join(self.root, f'{cate}_{split}_glbs')
            self.data_concept = os.path.join(self.root, f'{cate}_{split}_glbs')

        self.ids = []
        self.npz_list_real = []
        self.npz_list_concept = []
        for id in os.listdir(self.data_real):
            if id.split('_')[0] != 'chair':
                continue
            assert id.endswith(".npz")
            self.ids.append(id)
            self.npz_list_real.append(os.path.join(self.data_real, id))
            self.npz_list_concept.append(os.path.join(self.data_concept, id))
    
    def __len__(self) -> int:
        return len(self.ids)

    def _load_shape_from_occupancy_or_sdf(self, index: int, type: str):

        if type == 'real':
            data = np.load(f'{self.npz_list_real[index]}')
        elif type == 'concept':
            data = np.load(f'{self.npz_list_concept[index]}')
        coarse_surface = data["fps_coarse_surface"][:,0,:]
        sharp_surface = data["fps_sharp_surface"][:,0,:]
        ret = {
            "uid": self.ids[index],
            "coarse_surface": coarse_surface.astype(np.float32),
            "sharp_surface": sharp_surface.astype(np.float32),
        }

        return ret

    def _get_data(self, index):
        ret = {"uid": self.ids[index]}
        ret = self._load_shape_from_occupancy_or_sdf(index, type='real')

        ret_c = {"uid": self.ids[index]}
        ret_c = self._load_shape_from_occupancy_or_sdf(index, type='concept')

        return ret, ret_c
        
    def __getitem__(self, index):
        try:
            return self._get_data(index)
        except Exception as e:
            print(f"Error in {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

    def collate(self, batch):
        from torch.utils.data._utils.collate import default_collate_fn_map
        return torch.utils.data.default_collate(batch)