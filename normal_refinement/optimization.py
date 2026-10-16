"""Multi-view normal fitting with frequency separation and adaptive remeshing."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import trange

from .core.opt import MeshOptimizer
from .core.remesh import calc_vertex_normals
from .images import write_json
from .rendering import cameras, normalize_mesh, restore_mesh, save_views
from .utils.func import load_obj, save_obj
from .utils.render import NormalsRenderer


def gaussian_blur(images, kernel_size=121, sigma=20.):
    """Separable Gaussian, equivalent to the original 2D kernel with less memory."""
    coordinates = torch.arange(kernel_size, device=images.device, dtype=images.dtype) - kernel_size // 2
    kernel = torch.exp(-coordinates.square() / (2 * sigma * sigma))
    kernel /= kernel.sum()
    channels, pad = images.shape[1], kernel_size // 2
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    result = F.conv2d(F.pad(images, (pad, pad, 0, 0), mode='replicate'), horizontal, groups=channels)
    return F.conv2d(F.pad(result, (0, 0, pad, pad), mode='replicate'), vertical, groups=channels)


@torch.no_grad()
def make_targets(paths, reference, mv, frequency_separation=True, detail_strength=1.):
    target = torch.stack([torch.from_numpy(np.array(Image.open(path).convert('RGBA'), copy=True)).float() / 255 for path in paths]).to(reference.device)
    normal_camera = target[..., :3] * 2 - 1
    normal_world = torch.einsum('bhwj,bjk->bhwk', normal_camera, mv[:, :3, :3])
    target[..., :3] = (F.normalize(normal_world, dim=-1) + 1) / 2
    target[..., :3] *= target[..., 3:4]
    valid = (target[..., 3] > .1) & (reference[..., 3] > .1)
    if frequency_separation:
        clean = []
        for frame in target[..., :3].cpu().numpy():
            clean.append(torch.from_numpy(cv2.bilateralFilter((frame * 255).astype(np.uint8), 15, 10, 75)).float() / 255)
        clean = torch.stack(clean).to(reference.device)
        low_frequency = gaussian_blur(clean.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        detail = clean - low_frequency
        # Match the original 5x5 erosion repeated 8 times (effective width 33).
        mask = target[..., 3].unsqueeze(1)
        interior = (1 - F.max_pool2d(1 - mask, 33, stride=1, padding=16)).squeeze(1) > .5
        detail[~interior] = 0
        fused = reference[..., :3] + detail_strength * detail
        fused = (F.normalize(fused * 2 - 1, dim=-1) + 1) / 2
        target[..., :3] = torch.where(valid[..., None], fused, target[..., :3])
    target[..., :3] *= target[..., 3:4]
    return target


def normal_loss(images, target, mv, silhouette_weight=0.):
    overlap = (target[..., 3] > .1) & (images[..., 3] > .5)
    interior = (1 - F.max_pool2d(1 - overlap[:, None].float(), 11, stride=1, padding=5)).squeeze(1) > .5
    normal_camera = torch.einsum('bij,bhwj->bhwi', mv[:, :3, :3], images[..., :3] * 2 - 1)
    valid = interior & (normal_camera[..., 2] > np.cos(np.deg2rad(70)))
    if not valid.any():
        raise ValueError('No overlapping, front-facing pixels for normal fitting')
    normal = (images[..., :3][valid] - target[..., :3][valid]).square().mean()
    silhouette = (images[..., 3] - target[..., 3]).square().mean()
    return normal + silhouette_weight * silhouette, normal, silhouette, valid.sum()


def optimize(mesh_path, target_paths, output_path, output_dir, *, steps=101, resolution=2048,
             num_views=4, frequency_separation=True, detail_strength=1.,
             edge_min=.0005, edge_max=.002, laplacian_weight=.01, silhouette_weight=0.,
             debug_every=20):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    vertices, faces = load_obj(mesh_path)
    vertices, center, radius = normalize_mesh(vertices)
    mv, projection = cameras(num_views, vertices.device)
    renderer = NormalsRenderer(mv, projection, (resolution, resolution))
    with torch.no_grad():
        reference = renderer.render(vertices, calc_vertex_normals(vertices, faces), faces)
        target = make_targets(target_paths, reference, mv, frequency_separation, detail_strength)
        save_views(reference, directory / 'before_normals')
        save_views(target, directory / 'fitting_targets')
    optimizer = MeshOptimizer(vertices, faces, ramp=5, edge_len_lims=(edge_min, edge_max),
                              local_edgelen=True, laplacian_weight=laplacian_weight)
    vertices, faces = optimizer.remesh()
    history = []
    for step in trange(steps, desc='Normal refinement'):
        optimizer.zero_grad()
        normals = calc_vertex_normals(vertices, faces)
        images = renderer.render(vertices, normals, faces)
        loss, normal, silhouette, pixels = normal_loss(images, target, mv, silhouette_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite refinement loss at update {step}')
        loss.backward()
        if vertices.grad is None or not torch.isfinite(vertices.grad).all():
            raise FloatingPointError(f'Missing/nonfinite vertex gradient at update {step}')
        history.append({'step': step, 'loss': float(loss.detach()), 'normal_mse': float(normal.detach()),
                        'silhouette_mse': float(silhouette.detach()), 'valid_pixels': int(pixels),
                        'vertices': len(vertices), 'faces': len(faces)})
        optimizer.step()
        if debug_every and step % debug_every == 0:
            save_views(images, directory / 'iterations' / f'{step:04d}')
            write_json(directory / 'loss_history.json', history)
        # Release the render graph before topology updates allocate working buffers.
        del images, loss, normal, silhouette, normals
        vertices, faces = optimizer.remesh()
    with torch.no_grad():
        after = renderer.render(vertices, calc_vertex_normals(vertices, faces), faces)
        final_loss, final_normal, final_silhouette, pixels = normal_loss(after, target, mv, silhouette_weight)
        save_views(after, directory / 'after_normals')
        restored = restore_mesh(vertices, center, radius)
        if not torch.isfinite(restored).all() or not len(faces):
            raise FloatingPointError('Optimization produced invalid geometry')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_obj(restored, faces, output_path)
    write_json(directory / 'loss_history.json', history)
    metrics = {'updates': steps, 'initial_normal_mse': history[0]['normal_mse'],
               'final_normal_mse': float(final_normal), 'final_loss': float(final_loss),
               'final_silhouette_mse': float(final_silhouette), 'vertices': len(vertices), 'faces': len(faces),
               'normalization_center': center.cpu().tolist(), 'normalization_radius': float(radius),
               'output_coordinate_frame': 'same as input mesh'}
    write_json(directory / 'metrics.json', metrics)
    print(f'Saved refined mesh: {output_path}', flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh_path', required=True)
    parser.add_argument('--img_paths', nargs='+', required=True)
    parser.add_argument('--output_mesh_path', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--steps', type=int, default=101)
    parser.add_argument('--resolution', type=int, default=2048)
    parser.add_argument('--num_views', type=int, choices=[4, 6], default=4)
    parser.add_argument('--frequency_separation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--detail_strength', type=float, default=1.)
    parser.add_argument('--edge_min', type=float, default=.0005)
    parser.add_argument('--edge_max', type=float, default=.002)
    parser.add_argument('--laplacian_weight', type=float, default=.01)
    parser.add_argument('--silhouette_weight', type=float, default=0.)
    args = parser.parse_args()
    if len(args.img_paths) != args.num_views or args.steps < 1 or args.resolution < 64:
        parser.error('Need one target per view, positive steps and resolution >= 64')
    if not 0 < args.edge_min <= args.edge_max:
        parser.error('Require 0 < edge_min <= edge_max')
    optimize(args.mesh_path, args.img_paths, args.output_mesh_path, args.out_dir,
             steps=args.steps, resolution=args.resolution, num_views=args.num_views,
             frequency_separation=args.frequency_separation, detail_strength=args.detail_strength,
             edge_min=args.edge_min, edge_max=args.edge_max, laplacian_weight=args.laplacian_weight,
             silhouette_weight=args.silhouette_weight)


if __name__ == '__main__':
    main()
