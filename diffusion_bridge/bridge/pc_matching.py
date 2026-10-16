"""Align target VAE tokens to condition tokens using their FPS positions."""

import torch
from scipy.optimize import linear_sum_assignment


def emd_matching(target, condition):
    """Match coarse and sharp groups independently, preserving group order.

    Each input is (latents [B, Kc+Ks, C], coarse_xyz [B, Kc, 3],
    sharp_xyz [B, Ks, 3]). Both shapes must have the same group sizes.
    Return reordered target tokens and unchanged condition tokens.
    """
    target_latents, target_coarse, target_sharp = target
    condition_latents, condition_coarse, condition_sharp = condition
    batch_size, coarse_count, channels = (
        target_coarse.shape[0], target_coarse.shape[1], target_latents.shape[2]
    )
    if target_coarse.shape != condition_coarse.shape or target_sharp.shape != condition_sharp.shape:
        raise ValueError("Target and condition must have equal coarse/sharp group sizes")
    coarse_distances = torch.cdist(condition_coarse, target_coarse)
    sharp_distances = torch.cdist(condition_sharp, target_sharp)
    coarse_assignments = torch.empty(batch_size, coarse_count, dtype=torch.long, device=target_latents.device)
    sharp_assignments = torch.empty(batch_size, target_sharp.shape[1], dtype=torch.long, device=target_latents.device)
    for index in range(batch_size):
        # For square costs, SciPy returns rows in their original sorted order;
        # the selected columns map each condition position to a target token.
        _, coarse_columns = linear_sum_assignment(coarse_distances[index].cpu().numpy())
        _, sharp_columns = linear_sum_assignment(sharp_distances[index].cpu().numpy())
        coarse_assignments[index] = torch.from_numpy(coarse_columns).to(target_latents.device)
        sharp_assignments[index] = torch.from_numpy(sharp_columns).to(target_latents.device)
    coarse_indices = coarse_assignments.unsqueeze(-1).expand(-1, -1, channels)
    sharp_indices = sharp_assignments.unsqueeze(-1).expand(-1, -1, channels)
    coarse_tokens = torch.gather(target_latents[:, :coarse_count], 1, coarse_indices)
    sharp_tokens = torch.gather(target_latents[:, coarse_count:], 1, sharp_indices)
    return torch.cat([coarse_tokens, sharp_tokens], dim=1), condition_latents
