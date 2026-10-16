from normal_refinement.experimental.autoreg_opt import AutoregMeshOptimizer
from normal_refinement.utils.func import load_obj, make_star_cameras, normalize_vertices, save_obj, save_images
from tqdm import tqdm
import numpy as np
import os
import torch
try:
    from normal_refinement.utils.view import show
except:
    show = None


import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='mug_dragen_new')
    parser.add_argument('--mesh_path', type=str, default='data/mug_dragen_new.glb')
    parser.add_argument('--autoreg_mode', type=str, required=True, choices=['render', 'optimize', 'context'], help='Whether to render & mask, or optimize.')
    parser.add_argument('--view_idx', type=int, required=True, help='Current autoregressive step (0 to n_azimuth-1)')
    parser.add_argument('--n_azimuth', type=int, default=6, help='Total number of views around 360 degree.')
    parser.add_argument('--out_mask_path', type=str, default='', help='Path to save visibility mask for inpainting')
    parser.add_argument('--views', type=int, nargs='+', default=[0,1,2,3])
    parser.add_argument('--img_paths', type=str, nargs='+', default=[])
    parser.add_argument('--out_dir', type=str, default='./out_225')
    parser.add_argument('--output_mesh_path', type=str, default='result.obj')
    parser.add_argument('--use_frequency_separation', action='store_true', help='Use bilateral filtering to transfer details only, preserving ref global shape.')
    parser.add_argument('--force_subdivide', action='store_true', help='If True, forcefully subdivide the mesh to at least 4x vertices AND >10000 vertices.')
    parser.add_argument('--force_normalize', action='store_true', help='If True, forcefully normalize vertices and flip coordinate frame (only for the very first raw mesh)')
    args = parser.parse_args()

    steps = 101
    resolution = 2048
    decay_rate = 1

    # Specific views selection
    name = args.name
    selected_view_indices = args.views
    out_dir = args.out_dir

    # 1. Load initial object for alignment reference
    ref_vertices, ref_faces = load_obj(args.mesh_path)

    # 动态决定细分的目标顶点数
    original_vertex_count = ref_vertices.shape[0]
    target_vertex_count = 10000
    if args.force_subdivide:
        target_vertex_count = max(10000, original_vertex_count * 4)

    if ref_vertices.shape[0] < target_vertex_count:
        print(f"Mesh has {ref_vertices.shape[0]} vertices. Subdividing using pure PyTorch (Target: {target_vertex_count})...")

        while ref_vertices.shape[0] < target_vertex_count:
            F = ref_faces.shape[0]
            # 1. 提取所有边 (3条边/面)
            edges = torch.cat([ref_faces[:, [0, 1]], ref_faces[:, [1, 2]], ref_faces[:, [2, 0]]], dim=0)
            # 确保每条边的顶点按一致的顺序排列，以便找到唯一的无向边
            edges = torch.sort(edges, dim=1)[0]
            unique_edges, inverse_indices = torch.unique(edges, dim=0, return_inverse=True)

            # 2. 计算每条独特边的中点
            midpoints = (ref_vertices[unique_edges[:, 0]] + ref_vertices[unique_edges[:, 1]]) / 2.0

            # 3. 将中点追加到顶点张量中
            V = ref_vertices.shape[0]
            ref_vertices = torch.cat([ref_vertices, midpoints], dim=0)

            # 4. 获取每条边对应的新中点索引
            edge1_mid = V + inverse_indices[:F]
            edge2_mid = V + inverse_indices[F:2*F]
            edge3_mid = V + inverse_indices[2*F:]

            v0 = ref_faces[:, 0]
            v1 = ref_faces[:, 1]
            v2 = ref_faces[:, 2]

            # 5. 将每个原始面细分成 4 个新面 (Midpoint Subdivision)
            ref_faces = torch.cat([
                torch.stack([v0, edge1_mid, edge3_mid], dim=1),
                torch.stack([v1, edge2_mid, edge1_mid], dim=1),
                torch.stack([v2, edge3_mid, edge2_mid], dim=1),
                torch.stack([edge1_mid, edge2_mid, edge3_mid], dim=1)
            ], dim=0)

        print(f"After subdivision: {ref_vertices.shape[0]} vertices, {ref_faces.shape[0]} faces")

        # 保存细分后的网格
        subdivided_mesh_path = os.path.join(out_dir, name, 'subdivided_mesh.obj')
        os.makedirs(os.path.dirname(subdivided_mesh_path), exist_ok=True)
        save_obj(ref_vertices, ref_faces, subdivided_mesh_path)
        print(f"Subdivided mesh saved to: {subdivided_mesh_path}")

    # 避免多轮次自回归中对已经处理过的 mesh 重复归一化和翻转坐标系
    # 我们现在通过命令行明确指定是不是首次的原始网格
    if args.force_normalize:
        ref_vertices = normalize_vertices(ref_vertices)
        ref_vertices[..., [0, 2]] = - ref_vertices[..., [0, 2]]
        print("Normalizing and flipping base vertices...")

    # 2. Setup renderer (N views)
    distance = 10.0

    n_azimuth = args.n_azimuth
    _n_elevation = 1
    _, proj = make_star_cameras(1, 1, distance=distance) # 保留原始投影矩阵，后面手动生成球面螺旋 MV

    def get_spherical_cameras(n_views, device, dist=10.0):
        mvs = []
        # Spiral sampling: 多环螺旋上升/下降采样，保证连贯性
        # 第1~3个视角主视角（15度左右看轮廓），后几步过渡到俯视（45度）、底视（-20度）
        loops = max(1, n_views // 12)
        azimuths = np.linspace(0, 1.3 * np.pi * loops, n_views, endpoint=False)

        # 构造动态 elevation，从 +15度 升到 +45度 然后探到底部 -20度
        if n_views > 1:
            # e.g., for 6 views, elevations progress smoothly to cover top and bottom without extreme jump
            elevations = 15.0 + 25.0 * np.sin(np.linspace(0, np.pi * 1.5, n_views))
        else:
            elevations = np.array([15.0])

        elevations_rad = elevations * np.pi / 180.0

        for i in range(n_views):
            phi = torch.tensor(azimuths[i], device=device, dtype=torch.float32)
            theta = torch.tensor(elevations_rad[i], device=device, dtype=torch.float32)

            phi_rot = torch.eye(3, device=device)
            phi_rot[0, 0] = phi.cos()
            phi_rot[0, 2] = phi.sin()
            phi_rot[2, 0] = -phi.sin()
            phi_rot[2, 2] = phi.cos()

            theta_rot = torch.eye(3, device=device)
            theta_rot[1, 1] = theta.cos()
            theta_rot[1, 2] = -theta.sin()
            theta_rot[2, 1] = theta.sin()
            theta_rot[2, 2] = theta.cos()

            T = torch.eye(4, device=device)
            T[2, 3] = -dist

            mv_i = torch.eye(4, device=device)
            mv_i[:3, :3] = theta_rot @ phi_rot
            mvs.append(T @ mv_i)

        return torch.stack(mvs, dim=0)

    device_cam = 'cuda' if torch.cuda.is_available() else 'cpu'
    mv = get_spherical_cameras(n_azimuth, device_cam, dist=distance)

    if args.autoreg_mode in ['render', 'context']:
        selected_view_indices = [args.view_idx]
    else:
        selected_view_indices = args.views
    print(f"[{args.autoreg_mode.upper()} MODE] Run view_idx: {args.view_idx}/{args.n_azimuth}, Co-optimizing views: {selected_view_indices}")

    # 判断如果类别是 scissors，则将视角抬高 30 度（俯视）
    if 'scissors' in args.mesh_path:
        A = n_azimuth
        P = 1
        C = A * P
        device = mv.device if hasattr(mv, 'device') else 'cuda'

        phi = torch.arange(0, A, device=device) * (2 * torch.pi / A)
        phi_rot = torch.eye(3, device=device)[None, None].expand(A, 1, 3, 3).clone()
        phi_rot[:, 0, 2, 2] = phi.cos()
        phi_rot[:, 0, 2, 0] = -phi.sin()
        phi_rot[:, 0, 0, 2] = phi.sin()
        phi_rot[:, 0, 0, 0] = phi.cos()

        theta = torch.tensor([30.0 * torch.pi / 180.0], device=device)
        theta_rot = torch.eye(3, device=device)[None, None].expand(1, P, 3, 3).clone()
        theta_rot[0, :, 1, 1] = theta.cos()
        theta_rot[0, :, 1, 2] = -theta.sin()
        theta_rot[0, :, 2, 1] = theta.sin()
        theta_rot[0, :, 2, 2] = theta.cos()

        mv = torch.empty((C, 4, 4), device=device)
        mv[:] = torch.eye(4, device=device)
        mv[:, :3, :3] = (theta_rot @ phi_rot).reshape(C, 3, 3)

        trans = torch.eye(4, device=device)
        trans[2, 3] = -distance
        mv = trans @ mv

    # ===== AUTOREGRESSIVE MASK / FREEZE COMPUTATION =====
    # We use all views before args.view_idx to find "old" vertices
    distance = 10.0
    full_mv, full_proj = get_spherical_cameras(n_azimuth, device_cam, distance), proj

    # If scissors, patch full_mv (same as above)
    if 'scissors' in args.mesh_path:
        A = n_azimuth
        P = 1
        C = A * P
        device = full_mv.device if hasattr(full_mv, 'device') else 'cuda'
        phi = torch.arange(0, A, device=device) * (2 * torch.pi / A)
        phi_rot = torch.eye(3, device=device)[None, None].expand(A, 1, 3, 3).clone()
        phi_rot[:, 0, 2, 2] = phi.cos(); phi_rot[:, 0, 2, 0] = -phi.sin()
        phi_rot[:, 0, 0, 2] = phi.sin(); phi_rot[:, 0, 0, 0] = phi.cos()
        theta = torch.tensor([30.0 * torch.pi/180.0], device=device)
        theta_rot = torch.eye(3, device=device)[None, None].expand(1, P, 3, 3).clone()
        theta_rot[0, :, 1, 1] = theta.cos(); theta_rot[0, :, 1, 2] = -theta.sin()
        theta_rot[0, :, 2, 1] = theta.sin(); theta_rot[0, :, 2, 2] = theta.cos()
        full_mv = torch.empty((C, 4, 4), device=device)
        full_mv[:] = torch.eye(4, device=device)
        full_mv[:, :3, :3] = (theta_rot @ phi_rot).reshape(C, 3, 3)
        trans = torch.eye(4, device=device); trans[2, 3] = -distance
        full_mv = trans @ full_mv

    import nvdiffrast.torch as dr
    try:
        glctx = dr.RasterizeCudaContext()
    except AttributeError:
        glctx = dr.RasterizeGLContext()

    _V = ref_vertices.shape[0]
    _faces_i32 = ref_faces.type(torch.int32)
    vert_hom = torch.cat((ref_vertices, torch.ones(_V,1,device=ref_vertices.device)),axis=-1)

    old_faces_hit = set()
    # Render old views
    for v_old in range(args.view_idx):
        mvp = full_proj @ full_mv[v_old]
        if mvp.dim() == 2:
            mvp = mvp.unsqueeze(0)
        vertices_clip = (vert_hom @ mvp.transpose(-2,-1))
        if vertices_clip.dim() == 2:
            vertices_clip = vertices_clip.unsqueeze(0)
        rast_out, _ = dr.rasterize(glctx, vertices_clip, _faces_i32, resolution=[resolution, resolution], grad_db=False)
        hit = torch.unique(rast_out[0, ..., 3]).long() - 1
        old_faces_hit.update(hit[hit >= 0].tolist())

    old_faces_hit = torch.tensor(list(old_faces_hit), dtype=torch.long, device=ref_vertices.device)
    old_vertices = torch.unique(ref_faces[old_faces_hit]) if len(old_faces_hit) > 0 else torch.empty(0, dtype=torch.long, device=ref_vertices.device)

    frozen_mask = torch.zeros(_V, 1, dtype=torch.float32, device=ref_vertices.device)
    frozen_mask[old_vertices] = 1.0

    if args.autoreg_mode == 'context':
        from normal_refinement.core.remesh import calc_face_normals, calc_vertex_normals
        from normal_refinement.utils.render import NormalsRenderer
        import cv2

        f_normals = calc_face_normals(ref_vertices, ref_faces, normalize=True)
        v_normals = calc_vertex_normals(ref_vertices, ref_faces, f_normals)
        renderer = NormalsRenderer(full_mv, full_proj, image_size=(resolution, resolution))
        rend_color = renderer.render(ref_vertices, v_normals, ref_faces).cpu().numpy()

        small_size = 512
        resized_images = []
        for i in range(args.n_azimuth):
            img_i = rend_color[i]
            rgb = (img_i[..., :3] * 255).astype(np.uint8)[:,:,::-1] # BGR
            alpha = (img_i[..., 3:] * 255).astype(np.uint8)
            rgba = np.concatenate([rgb, alpha], axis=-1)
            rgba_small = cv2.resize(rgba, (small_size, small_size), interpolation=cv2.INTER_AREA)
            resized_images.append(rgba_small)

        stitched_rgba = np.concatenate(resized_images, axis=1) # Horizontal concat

        out_context_path = os.path.join(args.out_dir, "context_6_views.png")
        cv2.imwrite(out_context_path, stitched_rgba)
        print(f"[{args.autoreg_mode.upper()} MODE] Saved context grid to {out_context_path}")
        import sys; sys.exit(0)

    # If render mode, we output the mask for the current view!
    if args.autoreg_mode == 'render':
        curr_mvp = full_proj @ full_mv[args.view_idx]
        if curr_mvp.dim() == 2:
            curr_mvp = curr_mvp.unsqueeze(0)
        vertices_clip = (vert_hom @ curr_mvp.transpose(-2,-1))
        if vertices_clip.dim() == 2:
            vertices_clip = vertices_clip.unsqueeze(0)
        curr_rast, _ = dr.rasterize(glctx, vertices_clip, _faces_i32, resolution=[resolution, resolution], grad_db=False)

        not_frozen = 1.0 - frozen_mask
        interp_mask, _ = dr.interpolate(not_frozen.unsqueeze(0).contiguous(), curr_rast, _faces_i32)
        mask_2d = (interp_mask[0, ..., 0] > 0.5).float()

        # Save the mask (Pseudo mask style)
        # We ALSO need to render the normal map for Gemini to inpaint
        from normal_refinement.core.remesh import calc_face_normals, calc_vertex_normals
        from normal_refinement.utils.render import NormalsRenderer

        f_normals = calc_face_normals(ref_vertices, ref_faces, normalize=True)
        v_normals = calc_vertex_normals(ref_vertices, ref_faces, f_normals)
        renderer = NormalsRenderer(full_mv[args.view_idx:args.view_idx+1], full_proj, image_size=(resolution, resolution))
        rend_color = renderer.render(ref_vertices, v_normals, ref_faces)[0].cpu().numpy()

        # 转换为相机坐标系法线 (为了在每个视角下朝向相机的部分总是蓝紫色)
        # rend_color[..., :3] 是 World Space 的法线色彩 (0~1)
        world_normals = rend_color[..., :3] * 2.0 - 1.0
        R = full_mv[args.view_idx, :3, :3].cpu().numpy()
        # 已经验证过 Normal_world = Normal_cam @ R, 因此 Normal_cam = Normal_world @ R.T
        cam_normals = world_normals @ R.T

        # 重新归一化以防误差
        norms = np.linalg.norm(cam_normals, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-8
        cam_normals = cam_normals / norms

        cam_color = (cam_normals + 1.0) / 2.0
        cam_color = np.clip(cam_color * 255.0, 0, 255).astype(np.uint8)

        rgb = cam_color[..., ::-1] # RGB to BGR
        alpha = (rend_color[..., 3:] * 255).astype(np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=-1)

        # 构造 Pseudo Mask（带有物体的遮罩图）—— 改为使用规则矩形框覆盖
        mask_2d_np = mask_2d.cpu().numpy() > 0.5
        fg_mask = (alpha[..., 0] > 128)

        # 递归地将稀疏（如L型）的 Bounding Box 切分为多个更紧凑的小框
        def split_into_dense_bboxes(mask, x, y, w, h, min_density=0.4, min_size=200):
            if w <= min_size or h <= min_size:
                return [(x, y, w, h)]
            region = mask[y:y+h, x:x+w]
            pixels = np.count_nonzero(region)
            if pixels == 0: return []
            if pixels / float(w * h) >= min_density:
                return [(x, y, w, h)]

            boxes = []
            if w > h:
                half = w // 2
                left_region = region[:, :half]
                coords = np.argwhere(left_region)
                if len(coords) > 0:
                    ymin, xmin = coords.min(axis=0)
                    ymax, xmax = coords.max(axis=0)
                    boxes.extend(split_into_dense_bboxes(mask, x + xmin, y + ymin, xmax - xmin + 1, ymax - ymin + 1, min_density, min_size))
                right_region = region[:, half:]
                coords = np.argwhere(right_region)
                if len(coords) > 0:
                    ymin, xmin = coords.min(axis=0)
                    ymax, xmax = coords.max(axis=0)
                    boxes.extend(split_into_dense_bboxes(mask, x + half + xmin, y + ymin, xmax - xmin + 1, ymax - ymin + 1, min_density, min_size))
            else:
                half = h // 2
                top_region = region[:half, :]
                coords = np.argwhere(top_region)
                if len(coords) > 0:
                    ymin, xmin = coords.min(axis=0)
                    ymax, xmax = coords.max(axis=0)
                    boxes.extend(split_into_dense_bboxes(mask, x + xmin, y + ymin, xmax - xmin + 1, ymax - ymin + 1, min_density, min_size))
                bottom_region = region[half:, :]
                coords = np.argwhere(bottom_region)
                if len(coords) > 0:
                    ymin, xmin = coords.min(axis=0)
                    ymax, xmax = coords.max(axis=0)
                    boxes.extend(split_into_dense_bboxes(mask, x + xmin, y + half + ymin, xmax - xmin + 1, ymax - ymin + 1, min_density, min_size))
            return boxes

        pseudo_mask = np.zeros_like(rgb)
        # 填充已知区域的法线颜色
        pseudo_mask[fg_mask] = rgb[fg_mask]

        # 1. 形态学开运算：过滤掉散落在边缘的极小噪点
        mask_2d_uint8 = mask_2d_np.astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_2d_uint8, cv2.MORPH_OPEN, kernel)

        # 2. 寻找所有连通区域的轮廓，并分别为它们画局部的矩形框
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 按照连通区域的面积从大到小排序，最多只保留面积最大的 3 个连通分量
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        bbox_mask = np.zeros_like(mask_2d_np)
        pad = 20  # Padding 退回适度大小，避免各框融合

        final_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 过滤掉极小的局部噪点区域，不值得作为一个单独的 Inpainting Mask 框
            if w * h < 400:
                continue

            # 检查密度，拆分稀疏狂
            # 必须传入形态学清洗后的 mask_cleaned，防止切分时受边缘散碎零星像素的干扰
            dense_boxes = split_into_dense_bboxes(mask_cleaned > 0, x, y, w, h, min_density=0.4, min_size=200)
            final_boxes.extend(dense_boxes)

        for (bx, by, bw, bh) in final_boxes:
            y_min = max(0, by - pad)
            y_max = min(mask_2d_np.shape[0]-1, by + bh + pad)
            x_min = max(0, bx - pad)
            x_max = min(mask_2d_np.shape[1]-1, bx + bw + pad)

            # 将局部框内区域标记为需要涂白
            bbox_mask[y_min:y_max+1, x_min:x_max+1] = True

        # 在前景范围内，将组合出来的多个矩形框内涂白
        pseudo_mask[bbox_mask & fg_mask] = [255, 255, 255]

        import cv2
        if args.out_mask_path:
            cv2.imwrite(args.out_mask_path, pseudo_mask)
            print(f"[{args.autoreg_mode.upper()} MODE] Saved pseudo mask to {args.out_mask_path}")

        out_normal_path = os.path.join(args.out_dir, f"rendered_normal_{args.view_idx}.png")
        cv2.imwrite(out_normal_path, rgba)
        print(f"[{args.autoreg_mode.upper()} MODE] Saved rendered normal to {out_normal_path}")
        import sys; sys.exit(0)


    if selected_view_indices is not None and len(selected_view_indices) > 0:
        mv = mv[selected_view_indices]
        num_views = len(selected_view_indices)
        view_iter = selected_view_indices
        print(f"Refining ONLY on selected views: {selected_view_indices}")
    else:
        num_views = mv.shape[0]
        view_iter = range(num_views)
        print(f"Refining on ALL {num_views} views.")


    # Add Top and Bottom views
    def make_view(theta, phi, device):
        theta = torch.tensor(theta, device=device)
        phi = torch.tensor(phi, device=device)

        phi_rot = torch.eye(3, device=device)
        phi_rot[2,2] = phi.cos()
        phi_rot[2,0] = -phi.sin()
        phi_rot[0,2] = phi.sin()
        phi_rot[0,0] = phi.cos()

        theta_rot = torch.eye(3, device=device)
        theta_rot[1,1] = theta.cos()
        theta_rot[1,2] = -theta.sin()
        theta_rot[2,1] = theta.sin()
        theta_rot[2,2] = theta.cos()

        mv = torch.eye(4, device=device)
        mv[:3,:3] = theta_rot @ phi_rot

        # Translation (distance)
        T = torch.eye(4, device=device)
        T[2,3] = -distance

        return T @ mv

    _mv_top = make_view(np.pi/2, 0, mv.device)
    _mv_bottom = make_view(-np.pi/2, 0, mv.device)
    # mv = torch.cat([mv, mv_top.unsqueeze(0), mv_bottom.unsqueeze(0)], dim=0)

    renderer = NormalsRenderer(mv, proj, [resolution, resolution])

    # 3. Render initial object
    ref_normals = calc_vertex_normals(ref_vertices, ref_faces)
    ref_images = renderer.render(ref_vertices, ref_normals, ref_faces) # C, H, W, 4

    # 4. Load and process target images
    processed_targets = []

    # --- 引入对齐投影的方法 ---
    def project_normal_map(img, view_idx, save_dir):
        """
        将 img (BGRA/BGR) 的法线向量投影至球面上，生成修正后的 normal map，并返回修正后的图像。
        同时，保存修前和修正后的图片以备对比。
        """
        if img.shape[2] == 4:
            rgb = img[:, :, :3][..., ::-1].astype(np.float32) / 255.0
            mask = img[:, :, 3] > 0
        else:
            rgb = img[..., ::-1].astype(np.float32) / 255.0
            mask = np.any(img > 0, axis=-1)

        # 映射回 [-1, 1] 法线向量
        normals = rgb * 2.0 - 1.0
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)

        # 避免除 0
        norms[norms == 0] = 1e-8

        # 投影：归一化为单位向量，并映回 [0, 1] 和 [0, 255]
        normals_projected = normals / norms
        rgb_projected = (normals_projected + 1.0) / 2.0
        rgb_fixed = np.clip(rgb_projected * 255.0, 0, 255).astype(np.uint8)

        # 将结果复制回原始图片
        img_fixed = img.copy()
        img_fixed[mask, :3] = rgb_fixed[mask][..., ::-1]

        # --- Debug 图像保存 ---
        # 保存误差热力图
        error_map = np.abs(norms - 1.0)
        error_map_normalized = np.clip(error_map[..., 0] / 0.5, 0, 1)
        heatmap_uint8 = (error_map_normalized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color[~mask] = [0, 0, 0]

        os.makedirs(f'{save_dir}/0_projection_heatmaps/', exist_ok=True)
        cv2.imwrite(f'{save_dir}/0_projection_heatmaps/{view_idx}.png', heatmap_color)

        os.makedirs(f'{save_dir}/0_projected_fixed/', exist_ok=True)
        # 因为保存用，可以直接存 img_fixed
        cv2.imwrite(f'{save_dir}/0_projected_fixed/{view_idx}.png', img_fixed)

        return img_fixed

    for i, view_idx in enumerate(view_iter):
        if len(args.img_paths) > i:
            img_path = args.img_paths[i]
        else:
            img_path = f'./data_normals/{name}/{view_idx}.png'

        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found")

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.zeros((resolution, resolution, 4), dtype=np.uint8)

        # 1. Save Original
        os.makedirs(f'{out_dir}/{name}/debug_stages/1_original/', exist_ok=True)
        cv2.imwrite(f'{out_dir}/{name}/debug_stages/1_original/{view_idx}.png', img)
        # Background removal / Alpha (Perform before resize or after alignment? After is safer if alignment introduced padding)
        # The alignment naturally handles background if borderValue is 0.
        # But let's re-run the alpha logic just to be safe if original didn't have alpha
        if img.shape[2] == 3 or (img.shape[2] == 4 and np.all(img[:, :, 3] == 255)):
            print("Adding alpha channel and removing background")
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            try:
                import rembg
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgba = rembg.remove(img_rgb)
                img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA)
            except ImportError:
                print("rembg not installed, using simple background removal")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                # Simple background removal (assuming white/black background)
                # Heuristic: corner pixel is background
                bg_color = img[0, 0, :3]
                diff = np.sum(np.abs(img[:, :, :3] - bg_color), axis=2)
                mask = (diff > 20).astype(np.uint8) * 255
                img[:, :, 3] = mask

        # --- Alignment with Init Normals ---
        # Load corresponding init normal image for alignment
        # Note: ref_images is (N, H, W, 4) tensor.
        # We need numpy for OpenCV.
        ref_img_tensor = ref_images[i] # (H, W, 4)
        ref_img_np = (ref_img_tensor.detach().cpu().numpy() * 255).astype(np.uint8)

        # Use Alpha channel or Magnitude to find object mask for alignment
        # Target Mesh (Init) Mask
        if ref_img_np.shape[2] == 4:
            mask_ref = ref_img_np[:, :, 3]
        else:
            mask_ref = cv2.cvtColor(ref_img_np, cv2.COLOR_RGB2GRAY)
            _, mask_ref = cv2.threshold(mask_ref, 10, 255, cv2.THRESH_BINARY)

        # Input Image Mask (needs alpha handling first)
        # Temporary alpha extraction for alignment
        img_for_align = img.copy()
        if img_for_align.shape[2] == 3:
            # Simple bg removal for mask estimation
            bg_color = img_for_align[0, 0, :3]
            diff = np.sum(np.abs(img_for_align[:, :, :3] - bg_color), axis=2)
            mask_img = (diff > 20).astype(np.uint8) * 255
        else:
            mask_img = img_for_align[:, :, 3]

        # Calculate Moments
        M_ref = cv2.moments(mask_ref)
        M_img = cv2.moments(mask_img)

        if M_ref["m00"] > 0 and M_img["m00"] > 0:
            # Centroids
            cX_ref = int(M_ref["m10"] / M_ref["m00"])
            cY_ref = int(M_ref["m01"] / M_ref["m00"])
            cX_img = int(M_img["m10"] / M_img["m00"])
            cY_img = int(M_img["m01"] / M_img["m00"])

            # Scale (approximate by sqrt of area)
            scale = np.sqrt(M_ref["m00"] / M_img["m00"])

            # Construct Transformation Matrix
            # 1. Translate Image Center to Origin
            # 2. Scale
            # 3. Translate Origin to Ref Center

            T_to_origin = np.float32([[1, 0, -cX_img], [0, 1, -cY_img], [0, 0, 1]])
            S_matrix = np.float32([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
            T_to_ref = np.float32([[1, 0, cX_ref], [0, 1, cY_ref], [0, 0, 1]])

            M_final = T_to_ref @ S_matrix @ T_to_origin
            M_affine = M_final[:2, :] # 2x3 for warpAffine

            # Apply Warp
            h, w = resolution, resolution # Target sizex
            # Let's align the raw image to the canvas size of ref_images (which is resolution x resolution)
            img = cv2.warpAffine(img, M_affine, (resolution, resolution), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        else:
            # Fallback if moments fails (e.g. empty)
            img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)


        # Resize with padding (Square padding)
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if h != max_dim or w != max_dim:
            # Create a square canvas (transparent if 4 channels)
            square_img = np.zeros((max_dim, max_dim, img.shape[2]), dtype=img.dtype)
            # Calculate centering offsets
            x_off = (max_dim - w) // 2
            y_off = (max_dim - h) // 2
            square_img[y_off:y_off+h, x_off:x_off+w] = img
            img = square_img

        # Resize to target resolution
        if img.shape[0] != resolution or img.shape[1] != resolution:
            img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)

        # =============== 新增步骤：将输入图像色彩投影到单位球法线空间上 ===============
        debug_save_dir = f'{out_dir}/{name}/debug_stages/'
        img = project_normal_map(img, view_idx, debug_save_dir)
        # =========================================================================

        # Transform Camera-Space Normals to World-Space Normals
        # The input images are assumed to be in Camera-Space (Blue = facing camera).
        # The reference images are in World-Space.
        # We rotate the input normals using the inverse of the camera view rotation to align them.
        if img.shape[2] >= 3:
            # Check valid pixels
            check_mask = (img[..., 3] > 10) if img.shape[2] == 4 else np.ones(img.shape[:2], dtype=bool)
            if np.any(check_mask):
                # Get rotation R from View Matrix mv[i] (World -> Camera)
                # We want to apply the inverse rotation to the normals.
                # Normal_world = Normal_cam @ R (for row vectors)
                R = mv[i, :3, :3].detach().cpu().numpy()

                # Extract RGB from BGR, normalize to [-1, 1]
                # Valid pixels only
                valid_pixels = img[check_mask, :3]
                rgb = valid_pixels[..., ::-1].astype(np.float32) / 255.0
                normal_cam = rgb * 2.0 - 1.0

                # Rotate
                normal_world = normal_cam @ R

                # Back to color [0, 255]
                # rgb_world = ((normal_world + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
                rgb_world = ((normal_world + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)

                # Update image (BGR)
                img[check_mask, :3] = rgb_world[..., ::-1]

        # Convert BGRA/BGR to RGBA/RGB
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Save Aligned
        # Convert back to BGR for saving
        save_img = img.copy()
        if save_img.shape[2] == 4:
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGBA2BGRA)
        else:
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        os.makedirs(f'{out_dir}/{name}/debug_stages/2_aligned/', exist_ok=True)
        cv2.imwrite(f'{out_dir}/{name}/debug_stages/2_aligned/{view_idx}.png', save_img)

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float() / 255.0
        processed_targets.append(img_tensor)

    target_images = torch.stack([pt.to(ref_vertices.device) for pt in processed_targets])

    # --- Gaussian Blur Helper ---
    # Re-implementing Gaussian Blur using standard PyTorch ops to avoid dependencies
    # and for transparency.
    def get_gaussian_kernel(kernel_size=3, sigma=1.0, channels=3, device='cpu'):
        # Create 1D Gaussian kernel
        x_coord = torch.arange(kernel_size).to(device)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2. * torch.tensor([1., 1.]).to(device)
        variance = sigma**2.

        # Calculate G(x)
        # (x-u)^2 + (y-v)^2
        dist2 = torch.sum((xy_grid - mean)**2, dim=-1)
        kernel = (1./(2.*np.pi*variance)) * torch.exp(-dist2 / (2*variance))

        # Normalize
        kernel = kernel / torch.sum(kernel)

        # Reshape for conv2d: (out_channels, in_channels, kH, kW)
        # Grouped convolution: groups=channels, so in_channels per group is 1.
        # Weight shape should be (channels, 1, kH, kW)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)

        return kernel

    def gaussian_blur_tensor(img_tensor, kernel_size=5, sigma=1.0):
        # img_tensor: (B, C, H, W)
        b, c, h, w = img_tensor.shape
        kernel = get_gaussian_kernel(kernel_size, sigma, c, img_tensor.device)

        # Padding
        pad = (kernel_size - 1) // 2

        # Apply with replicate padding to avoid dark edges (Halo effect)
        img_padded = torch.nn.functional.pad(img_tensor, (pad, pad, pad, pad), mode='replicate')
        blurred = torch.nn.functional.conv2d(img_padded, kernel, padding=0, groups=c)

        return blurred

    # --- Pre-processing: Denoise Target Images ---
    # Analyze the difference map and remove small, isolated noise specs using Morphological Opening.
    # Also handle global color shift (bias) between Target and Init.

    print("Running Spatial Denoising on Target Images...")
    with torch.no_grad():
        # --- Advanced Frequency Separation (Detail Transfer) ---
        # Improved: Use Bilateral Filtering (Edge-Preserving Smoothing) instead of Gaussian Blur
        # Logic:
        # 1. Trust Ref for Low-Freq (Shape/Geometry)
        # 2. Trust Target for High-Freq (Texture/Details)

        use_frequency_separation = args.use_frequency_separation
        if use_frequency_separation:
            # print("Applying Frequency Separation (Bilateral) to transfer details only...")
            # 1. Define ROI
            if target_images.shape[-1] == 4 and ref_images.shape[-1] == 4:
                mask_target = target_images[..., 3] > 0.1
                mask_ref = ref_images[..., 3] > 0.1
                valid_region = mask_target & mask_ref
            else:
                valid_region = torch.ones(target_images.shape[:-1], dtype=torch.bool, device=target_images.device)

            tgt_np = (target_images[..., :3].detach().cpu().numpy() * 255).astype(np.uint8)

            target_clean_list = []

            # Bilateral Parameters (Moderate settings to denoise but keep scales)
            # d=15: Local neighborhood enough for small scale smoothing
            # sigmaColor=25: Strict enough to preserve sharp scale edges
            # sigmaSpace=75: Smooth out flat areas
            d = 15
            sigmaColor = 10  # Keep rigid edges
            sigmaSpace = 75

            for i in range(tgt_np.shape[0]):
                # Apply bilateral filter per image to get "Digital Clean" version
                # This removes pixel noise but KEEPS scaler structure and global shape
                blur = cv2.bilateralFilter(tgt_np[i], d, sigmaColor, sigmaSpace)
                target_clean_list.append(torch.from_numpy(blur).float() / 255.0)

            # target_clean = target_images[..., :3] # Bypass bilateral for now, use original target as "clean" version for structure extraction. We will rely on the mid-frequency extraction to separate scales from noise.
            target_clean = torch.stack(target_clean_list).to(target_images.device)

            # # [DEBUG SAVE] 1. 保存双边滤波后的干净目标图
            os.makedirs(f'{out_dir}/{name}/debug_stages/freq_1_target_clean/', exist_ok=True)
            save_images(target_clean, f'{out_dir}/{name}/debug_stages/freq_1_target_clean/')

            # 3. Hybrid Detail Extraction Strategy (The "Best of Both Worlds")
            # Step A: Extract "Mid-Frequency" details (Scales) from the Clean Target
            # We blur the Clean Target to remove the scales, finding its underlying shape.
            tgt_clean_nchw = target_clean.permute(0, 3, 1, 2)

            # Kernel large enough to blur out individual scales (~40-60 px)
            k_size_structure = 121
            sigma_structure = 20

            # Ensure kernel size is odd
            if k_size_structure % 2 == 0: k_size_structure += 1

            try:
                 target_structure = gaussian_blur_tensor(tgt_clean_nchw, kernel_size=k_size_structure, sigma=sigma_structure)
            except Exception as e:
                 print(f"Gaussian Blur failed: {e}, falling back to clean target.")
                 target_structure = tgt_clean_nchw

            target_structure_nhwc = target_structure.permute(0, 2, 3, 1)

            # [DEBUG SAVE] 2. 保存高斯模糊后的低频结构图 (Base Geometry)
            os.makedirs(f'{out_dir}/{name}/debug_stages/freq_2_target_structure/', exist_ok=True)
            save_images(target_structure_nhwc, f'{out_dir}/{name}/debug_stages/freq_2_target_structure/')

            # Mid-Freq Details = Clean Target (with scales) - Blurred Target (no scales)
            # This contains ONLY the scales, centred around 0, without the global bias/shape of the Target.
            mid_freq_details = target_clean - target_structure_nhwc

            # --- 解决边缘噪声问题 ---
            # 边缘处（尤其是轮廓边缘）由于高斯模糊和背景色（通常是黑色或透明）的混合，
            # 会产生非常强烈的假梯度（False Edges），这些不是真实的表面细节。
            # 我们需要像之前计算 Loss 时一样，利用 target_images 的 Alpha 通道进行向内腐蚀，
            # 把边缘一圈的 mid_freq_details 强行清零。

            # 将 target 的 mask 取出并转为 numpy 处理腐蚀
            detail_val_mask_np = (target_images[..., 3] > 0.1).detach().cpu().numpy().astype(np.uint8)
            eroded_detail_mask_np = np.zeros_like(detail_val_mask_np)

            # 使用一个较大的核来腐蚀边缘，因为高斯模糊的核（k_size=61）很大，
            # 边缘效应的影响范围也会比较宽。这里可以使用至少与高斯 sigma 相关的腐蚀范围。
            # 比如腐蚀迭代次数稍微大一点
            detail_kernel = np.ones((5, 5), np.uint8)
            # padding 效应范围大约是 kernel_size // 2，所以这里腐蚀几次把这个边缘吃掉
            # 也可以根据分辨率动态调整
            detail_erosion_iter = 8

            for e_idx in range(detail_val_mask_np.shape[0]):
                eroded_detail_mask_np[e_idx] = cv2.erode(detail_val_mask_np[e_idx], detail_kernel, iterations=detail_erosion_iter)

            eroded_detail_mask = torch.from_numpy(eroded_detail_mask_np).bool().to(target_images.device)

            # 将边缘部分的细节清零
            mid_freq_details[~eroded_detail_mask] = 0.0
            # ------------------------

            # [DEBUG SAVE] 3. 保存提取出的中频细节图 (原始值，包含负数，保存时可能会被截断到 [0, 1])
            os.makedirs(f'{out_dir}/{name}/debug_stages/freq_3_mid_freq_details/', exist_ok=True)
            save_images(mid_freq_details.abs().clamp(0, 1), f'{out_dir}/{name}/debug_stages/freq_3_mid_freq_details/')

            # Step B: Fuse onto Reference
            # Final = Reference (perfect geometry) + Mid-Freq Details (Clean Scales)

            # 1. 在颜色空间（线性）做加法
            strength = 1 # Strength of the detail transfer, can be tuned
            fused_target_rgb = ref_images[..., :3] + mid_freq_details * strength

            # 2. 转换到真正的法线向量空间 [-1, 1]
            fused_normals = fused_target_rgb * 2.0 - 1.0

            # 4. 投影到单位球面上 (L2 Normalize)，并附带极小值防御防止除 0
            fused_normals_normalized = torch.nn.functional.normalize(fused_normals, p=2, dim=-1, eps=1e-8)

            # 5. 换算回 [0, 1] RGB颜色空间，做最终的安全 clamp 防御浮点溢出
            processed_target = ((fused_normals_normalized + 1.0) / 2.0).clamp(0, 1)


            # [DEBUG SAVE] 4. 保存融合细节后的最终目标图
            os.makedirs(f'{out_dir}/{name}/debug_stages/freq_4_fused_target/', exist_ok=True)
            save_images(processed_target, f'{out_dir}/{name}/debug_stages/freq_4_fused_target/')

            # Apply details globally across all views; rely on multi-view loss to reconcile
            target_images[..., :3] = torch.where(
                valid_region.unsqueeze(-1),
                processed_target,
                target_images[..., :3] # Keep original background
            )

            print("Hybrid Frequency Separation Done: Ref Base + Bilateral Details.")

    save_images(target_images, f'{out_dir}/{name}/target_images_processed/')
    # ---------------------------------------------

    save_images(target_images, f'{out_dir}/{name}/target_images/')
    save_images(ref_images, f'{out_dir}/{name}/init_normals/')

    vertices, faces = ref_vertices, ref_faces
    # vertices,faces = make_sphere(level=2,radius=.5)
    print(vertices.shape,faces.shape)

    # vertices, faces = load_obj('data/sofa_concept.glb')
    # vertices = normalize_vertices(vertices)

    # 2. init from coarse mesh
    relax_mask = torch.zeros_like(frozen_mask) # 放弃顶点硬性冻结，改为使用历史视图联合优化的软性软约束！
    opt = AutoregMeshOptimizer(vertices, faces, frozen_mask=relax_mask, ramp=5, edge_len_lims=(0.001, 0.004), local_edgelen=True, laplacian_weight=0.01) # 0.02,  0.005,0.020
    # opt = MeshOptimizer(vertices, faces)

    vertices = opt.vertices
    _snapshots = []

    R_batch = mv[:, :3, :3] # (Num_views, 3, 3) On GPU

    # ==== Coarse-to-Fine Strategy Parameters ====
    start_edge_len_lims = (0.01, 0.04)  # 初始化阶段使用较大边长(粗糙)，加速大结构收敛
    end_edge_len_lims = (0.0025, 0.005)   # 结束阶段使用较小边长(精细)，雕刻高频法线细节

    # remesh before optimization to ensure good initial vertex distribution
    vertices,faces = opt.remesh()
    save_obj(vertices,faces, f'{out_dir}/{name}/remeshed_init.obj')

    for i in tqdm(range(steps)):
        # --- 动态收紧边长限制 (Linear decay) ---
        progress = i / max(1, steps - 1)
        current_min = start_edge_len_lims[0] + progress * (end_edge_len_lims[0] - start_edge_len_lims[0])
        current_max = start_edge_len_lims[1] + progress * (end_edge_len_lims[1] - start_edge_len_lims[1])

        opt._edge_len_lims = (current_min, current_max)
        # 强制将现有顶点的期望边长夹逼到新的范围内
        if hasattr(opt, '_ref_len') and opt._ref_len is not None:
            opt._ref_len.clamp_(*opt._edge_len_lims)
        # ----------------------------------------

        opt.zero_grad()
        opt._lr *= decay_rate
        normals = calc_vertex_normals(vertices,faces)
        images = renderer.render(vertices,normals,faces)

        mask = target_images[..., -1] > 0.1
        d_mask = images[..., -1] > 0.5
        both_mask = mask & d_mask

        # ================= 性能改进 1：全 GPU 原生腐蚀算法 =================
        # 丢弃 cv2.erde，使用负向的最大池化 (Max Pooling) 在 GPU 上直接执行形态学腐蚀。
        # cv2 迭代 5 次的 3x3 腐蚀，等效于一个 Kernel size = 11 的大核 (2*iterations + 1)

        # 转为 (B, C, H, W) 浮点格式
        both_mask_float = both_mask.unsqueeze(1).float()
        # 翻转掩码（白变黑，黑变白），因为最大池化会让高亮区域变大（也就是让黑色部分侵蚀白色）
        inverted_mask = 1.0 - both_mask_float
        # 在 GPU 上执行高速池化展开侵蚀
        eroded_inv = torch.nn.functional.max_pool2d(inverted_mask, kernel_size=11, stride=1, padding=5)
        # 再翻回来，得到腐蚀后的 Mask
        erosion_mask = (1.0 - eroded_inv) > 0.5
        erosion_mask = erosion_mask.squeeze(1) # 回到 (B, H, W) 的 Bool Tensor

        # ===================================================================

        # 3. 角度掩码 (Angle Filter): 忽略表面法线与相机夹角接近 90 度的区域 (掠射角)
        world_normals = images[..., :3] * 2.0 - 1.0

        # 设定允许的最大夹角 60 度。
        angle_thresh_deg = 70.0
        cos_thresh = float(np.cos(np.radians(angle_thresh_deg)))

        # ================= 性能改进 2：批量矩阵乘法 (Batched MatMul) =================
        # 替换原本的 for 循环，使用 GPU 原生 bmm 并发计算所有视角的相机坐标系法线
        # (B, H*W, 3) @ (B, 3, 3) -> (B, H*W, 3)
        B = world_normals.shape[0]
        wn_flat = world_normals.view(B, resolution * resolution, 3)
        n_cam_flat = torch.bmm(wn_flat, R_batch.transpose(1, 2))
        n_cam = n_cam_flat.view(B, resolution, resolution, 3)

        n_cam_z = n_cam[..., 2]
        angle_mask = n_cam_z > cos_thresh
        # ============================================================================

        final_mask = erosion_mask & angle_mask

        # 赋予历史视角更高权重，软性锚定已优化的几何
        B_views = images.shape[0]
        weight_tensor = torch.ones(B_views, device=images.device)
        if B_views > 1:
            weight_tensor[:-1] = 5.0  # 历史视角权重
            weight_tensor[-1] = 1.0   # 当前新增视角权重

        # 1. 表面法线损失 (仅在网格及目标共有的有效区域内计算)
        diff_vec = images[..., :3] - target_images[..., :3]
        sq_diff = diff_vec.pow(2) * weight_tensor.view(-1, 1, 1, 1)
        loss_normal = sq_diff[final_mask].mean()

        # 2. 引入轮廓损失 (Silhouette Loss) - 允许网格向外扩张或向内收缩
        sil_diff = (images[..., -1] - target_images[..., -1]).pow(2)
        sil_diff = sil_diff * weight_tensor.view(-1, 1, 1)
        loss_sil = sil_diff.mean()

        # 合并 Loss (sil_weight 一般设为 1.0 或 2.0，视情况可微调)
        sil_weight = 1.0
        loss = loss_normal + sil_weight * loss_sil

        loss.backward()

        if vertices.grad is None:
            vertices.grad = torch.zeros_like(vertices)

        opt.step()

        if i % 20 == 0:
            with torch.no_grad():
                diff_rgb = torch.zeros_like(diff_vec)

                # Apply mask: Only show diff where loss is computed
                # We use boolean masking.
                diff_rgb[final_mask] = diff_vec[final_mask].abs()

                diff_np = diff_rgb.detach().cpu().numpy()

                save_path = f'{out_dir}/{name}/diff_heatmaps/{i}/'
                normals_save_path = f'{out_dir}/{name}/intermediate_normals/{i}/'
                os.makedirs(save_path, exist_ok=True)
                os.makedirs(normals_save_path, exist_ok=True)

                # 同时也处理渲染出的当前网格法线图，把它从 [B, H, W, 4] 变成可视化的 numpy array
                images_np = images.detach().cpu().numpy()

                for v_idx in range(diff_np.shape[0]):
                    # 处理差异图 (Diff heatmap)
                    d_img = (np.clip(diff_np[v_idx], 0, 1) * 255).astype(np.uint8)
                    d_img_bgr = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
                    valid_mask_np = final_mask[v_idx].detach().cpu().numpy()
                    d_img_bgr[~valid_mask_np] = 0
                    cv2.imwrite(os.path.join(save_path, f'{v_idx}.png'), d_img_bgr)

                    # 处理当前 mesh 的法线图 (Current Normals)
                    # 提取 RGB 颜色并转换
                    norm_img = (np.clip(images_np[v_idx, ..., :3], 0, 1) * 255).astype(np.uint8)
                    norm_img_bgr = cv2.cvtColor(norm_img, cv2.COLOR_RGB2BGR)

                    # 如果有 Alpha 通道，加上背景变黑处理
                    if images_np.shape[-1] == 4:
                        alpha_mask = images_np[v_idx, ..., 3] > 0.5
                        norm_img_bgr[~alpha_mask] = 0

                    cv2.imwrite(os.path.join(normals_save_path, f'{v_idx}.png'), norm_img_bgr)

        vertices,faces = opt.remesh()

    save_obj(vertices,faces,args.output_mesh_path)
    save_images(images, f'{out_dir}/{name}/images/')

    # === 新增：保存当前视角优化后的法线图 (Camera Space) 会与输入 Gemini 时渲染的图格式完全对齐 ===
    if args.autoreg_mode == 'optimize':
        with torch.no_grad():
            final_normals = calc_vertex_normals(vertices, faces)
            curr_mv = full_mv[args.view_idx:args.view_idx+1].to(vertices.device)
            renderer_single = NormalsRenderer(curr_mv, proj, image_size=(resolution, resolution))
            rend_color_opt = renderer_single.render(vertices, final_normals, faces)[0].cpu().numpy()

            world_normals_opt = rend_color_opt[..., :3] * 2.0 - 1.0
            R_opt = curr_mv[0, :3, :3].cpu().numpy()
            cam_normals_opt = world_normals_opt @ R_opt.T

            norms_opt = np.linalg.norm(cam_normals_opt, axis=-1, keepdims=True)
            norms_opt[norms_opt == 0] = 1e-8
            cam_normals_opt = cam_normals_opt / norms_opt

            cam_color_opt = (cam_normals_opt + 1.0) / 2.0
            cam_color_opt = np.clip(cam_color_opt * 255.0, 0, 255).astype(np.uint8)

            rgb_opt = cam_color_opt[..., ::-1] # RGB -> BGR格式
            alpha_opt = (rend_color_opt[..., 3:] * 255).astype(np.uint8)
            rgba_opt = np.concatenate([rgb_opt, alpha_opt], axis=-1)

            opt_out_path = os.path.join(args.out_dir, f"optimized_normal_{args.view_idx}.png")
            cv2.imwrite(opt_out_path, rgba_opt)
            print(f"[{args.autoreg_mode.upper()} MODE] Saved optimized normal (camera space) to {opt_out_path} for comparison.")

if __name__ == "__main__":
    main()
