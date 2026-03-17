import torch
from scipy.optimize import linear_sum_assignment

def emd_matching(batch, cond):
    gt_latents, gt_coarse_pc, gt_sharp_pc = batch
    cond_latents, cond_coarse_pc, cond_sharp_pc = cond
    
    B, N, C = gt_coarse_pc.shape[0], gt_coarse_pc.shape[1], gt_latents.shape[2]
    
    # 计算距离矩阵
    coarse_dist_matrix = torch.cdist(cond_coarse_pc, gt_coarse_pc)  # [B, N, N]
    sharp_dist_matrix = torch.cdist(cond_sharp_pc, gt_sharp_pc)    # [B, M, M]
    
    # d'd始化匹配索引
    coarse_assignments = torch.zeros(B, N, dtype=torch.long, device=gt_latents.device)
    sharp_assignments = torch.zeros(B, gt_sharp_pc.shape[1], dtype=torch.long, device=gt_latents.device)
    
    # 对每个batch进行EMD匹配
    for b in range(B):
        # Coarse点云的EMD匹配
        coarse_cost_matrix = coarse_dist_matrix[b].cpu().numpy()
        coarse_row_ind, coarse_col_ind = linear_sum_assignment(coarse_cost_matrix)
        coarse_assignments[b] = torch.from_numpy(coarse_col_ind).to(gt_latents.device)
        
        # Sharp点云的EMD匹配
        sharp_cost_matrix = sharp_dist_matrix[b].cpu().numpy()
        sharp_row_ind, sharp_col_ind = linear_sum_assignment(sharp_cost_matrix)
        sharp_assignments[b] = torch.from_numpy(sharp_col_ind).to(gt_latents.device)
    
    # 扩展索引以匹配latent维度
    coarse_idx_exp = coarse_assignments.unsqueeze(-1).expand(-1, -1, C)
    sharp_idx_exp = sharp_assignments.unsqueeze(-1).expand(-1, -1, C)
    
    # 根据EMD匹配结果重排序latents
    gt_latent_coarse_sorted = torch.gather(gt_latents[:, :N, :], 1, coarse_idx_exp)
    gt_latent_sharp_sorted = torch.gather(gt_latents[:, N:, :], 1, sharp_idx_exp)
    
    # 拼接coarse和sharp的排序结果
    gt_latents_sorted = torch.cat([gt_latent_coarse_sorted, gt_latent_sharp_sorted], dim=1)
    
    return gt_latents_sorted, cond_latents

def closest_point_matching(batch, cond):
    gt_latents, gt_coarse_pc, gt_sharp_pc = batch
    cond_latents, cond_coarse_pc, cond_sharp_pc = cond

    B, N, C = gt_coarse_pc.shape[0], gt_coarse_pc.shape[1], gt_latents.shape[2]

    coarse_dist_matrix = torch.cdist(cond_coarse_pc, gt_coarse_pc)
    coarse_nn_idx = torch.argmin(coarse_dist_matrix, dim=2)

    sharp_dist_matrix = torch.cdist(cond_sharp_pc, gt_sharp_pc)
    sharp_nn_idx = torch.argmin(sharp_dist_matrix, dim=2)

    coarse_idx_exp = coarse_nn_idx.unsqueeze(-1).expand(-1, -1, C)
    sharp_idx_exp = sharp_nn_idx.unsqueeze(-1).expand(-1, -1, C)

    # coarse 排序
    gt_latent_coarse_sorted = torch.gather(gt_latents[:, :N, :], 1, coarse_idx_exp)
    # sharp 排序
    gt_latent_sharp_sorted = torch.gather(gt_latents[:, N:, :], 1, sharp_idx_exp)

    gt_latents_sorted = torch.cat([gt_latent_coarse_sorted, gt_latent_sharp_sorted], dim=1)

    return gt_latents_sorted, cond_latents