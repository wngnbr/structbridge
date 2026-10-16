"""Shared camera ordering and normal coordinates for rendering and fitting."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .core.remesh import calc_vertex_normals
from .utils.func import load_obj, make_star_cameras
from .utils.render import NormalsRenderer


def cameras(num_views=4, device='cuda'):
    """Four azimuth views, then top and bottom; row-major atlas ordering."""
    mv, projection = make_star_cameras(4, 1, distance=10., device=device)
    if num_views == 6:
        extra = []
        for theta in (np.pi / 2, -np.pi / 2):
            view = torch.eye(4, device=device)
            view[1, 1] = view[2, 2] = np.cos(theta)
            view[1, 2], view[2, 1] = -np.sin(theta), np.sin(theta)
            view[2, 3] = -10.
            extra.append(view)
        mv = torch.cat([mv, torch.stack(extra)])
    elif num_views != 4:
        raise ValueError('num_views must be 4 or 6')
    return mv, projection


def normalize_mesh(vertices):
    """Return optimizer coordinates and the inverse transform (no input mutation)."""
    center = (vertices.amin(0) + vertices.amax(0)) / 2
    centered = vertices - center
    radius = centered.norm(dim=-1).max()
    if not torch.isfinite(radius) or radius <= 0:
        raise ValueError('Mesh has zero or nonfinite extent')
    flip = vertices.new_tensor([-1., 1., -1.])
    return centered / radius * flip, center, radius


def restore_mesh(vertices, center, radius):
    return vertices * vertices.new_tensor([-1., 1., -1.]) * radius + center


def save_views(images, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(images.detach().cpu().numpy()):
        path = directory / f'{index:02d}.png'
        Image.fromarray((np.clip(frame, 0, 1) * 255).round().astype(np.uint8)).save(path)
        paths.append(path)
    return paths


def merge_views(paths, destination):
    columns = 2 if len(paths) == 4 else 3
    with Image.open(paths[0]) as first:
        width, height = first.size
    atlas = Image.new('RGB', (width * columns, height * 2), 'black')
    for index, path in enumerate(paths):
        with Image.open(path) as frame:
            atlas.paste(frame.convert('RGB'), (index % columns * width, index // columns * height))
    atlas.save(destination)


@torch.no_grad()
def render_normals(mesh_path, output_dir, resolution=1024, num_views=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces = load_obj(mesh_path)
    vertices, _, _ = normalize_mesh(vertices)
    mv, projection = cameras(num_views, vertices.device)
    renderer = NormalsRenderer(mv, projection, (resolution, resolution))
    world = renderer.render(vertices, calc_vertex_normals(vertices, faces), faces)
    camera_normals = F.normalize(torch.einsum('bij,bhwj->bhwi', mv[:, :3, :3], world[..., :3] * 2 - 1), dim=-1)
    alpha = world[..., 3:4]
    rgb = (camera_normals + 1) / 2 * alpha
    paths = save_views(torch.cat([rgb, alpha], dim=-1), output_dir / 'rendered_views')
    atlas = output_dir / 'rendered_normals.png'
    merge_views(paths, atlas)
    return atlas
