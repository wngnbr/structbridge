import torch
from . import dist_util


class BatchedSeedGenerator:

    def __init__(self, seeds=None):
        self.num_samples = len(seeds)
        if torch.cuda.is_available():
            self.rng = [torch.Generator(dist_util.dev()) for _ in range(self.num_samples)]
        else:
            self.rng = [torch.Generator() for _ in range(self.num_samples)]
        [rng.manual_seed(int(seeds[i])) for i, rng in enumerate(self.rng)]

    def randn(self, size, dtype=torch.float, device="cpu"):
        assert size[0] == self.num_samples
        return torch.cat(
            [
                torch.randn(1, *size[1:], generator=self.rng[i], dtype=dtype, device=device)
                for i in range(self.num_samples)
            ],
            dim=0,
        )

    def randint(self, low, high, size, dtype=torch.long, device="cpu"):
        assert size[0] == self.num_samples
        return torch.cat(
            [
                torch.randint(
                    low,
                    high,
                    generator=self.rng[i],
                    size=(1, *size[1:]),
                    dtype=dtype,
                    device=device,
                )
                for i in range(self.num_samples)
            ],
            dim=0,
        )

    def randn_like(self, tensor):
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(size, dtype=dtype, device=device)
