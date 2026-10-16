"""Paired surface points for concept-to-real shape translation."""

import hashlib
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class PairedShapeDataset(Dataset):
    def __init__(self, root, category="all", category_filter="chair"):
        self.real_dir = Path(root) / f"{category}_real_glbs"
        self.concept_dir = Path(root) / f"{category}_concept_glbs"
        self.ids = sorted(path.name for path in self.real_dir.glob("*.npz")
                          if not category_filter or path.name.split("_")[0] == category_filter)
        if not self.ids:
            raise ValueError(f"No paired shapes in {self.real_dir} for {category_filter!r}")
        missing = [name for name in self.ids if not (self.concept_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing concept pairs in {self.concept_dir}: {missing[:5]}")
        self.fingerprint = hashlib.sha256("\n".join(self.ids).encode()).hexdigest()

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def load_surface(path):
        """Return coarse/sharp arrays [N, 6] with XYZ coordinates and normals."""
        with np.load(path, allow_pickle=False) as data:
            return {
                "uid": path.name,
                "coarse_surface": data["fps_coarse_surface"][:, 0, :].astype(np.float32),
                "sharp_surface": data["fps_sharp_surface"][:, 0, :].astype(np.float32),
            }

    def __getitem__(self, index):
        name = self.ids[index]
        return self.load_surface(self.real_dir / name), self.load_surface(self.concept_dir / name)
