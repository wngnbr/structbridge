"""File-based image generation handoff and camera-normal target preparation."""

import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image


class ImageRequired(RuntimeError):
    """A valid generation request exists, but no generated image is available."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def generation_prompt(category, num_views):
    layout = '2x2' if num_views == 4 else '3 columns by 2 rows'
    return (
        f'Edit this {layout} atlas of {num_views} camera-space normal maps of the same {category}. '
        'Add clean, subtle, category-appropriate geometric surface details. Preserve the exact '
        'view ordering, framing, positions, scale, silhouette, holes and structural parts in every view. '
        'Keep the black background and normal encoding RGB=(camera-space normal+1)/2; '
        'front-facing surfaces remain blue-violet. Keep details consistent across views. '
        'Remove surface noise without changing global curvature. Do not add lighting, materials, '
        'shadows, photographic colors, text, borders or new parts. Return one PNG with exactly '
        'the same aspect ratio and grid as the input. This is geometry supervision, not a beauty render.'
    )


def require_generated_image(reference, output_dir, category, num_views, generated=None, wait_seconds=0,
                            image_provider='manual'):
    if image_provider not in ('auto', 'gemini', 'manual'):
        raise ValueError(f'Unknown image provider: {image_provider}')
    directory = Path(output_dir)
    target = Path(generated).resolve() if generated else directory / 'generated_normals.png'
    request = {
        'version': 1, 'status': 'image_required', 'provider': 'external',
        'reference_image': str(Path(reference).resolve()), 'reference_sha256': sha256(reference),
        'output_image': str(target.resolve()), 'category': category, 'num_views': num_views,
        'prompt': generation_prompt(category, num_views),
        'instructions': 'Use the reference as an edit target; save the completed image atomically to output_image, then rerun the same command.',
    }
    write_json(directory / 'image_request.json', request)
    (directory / 'image_prompt.txt').write_text(request['prompt'] + '\n')
    # An explicit path always means a user-supplied image, including when missing.
    # Never replace it with an API result or regenerate a cached image implicitly.
    if not target.is_file() and generated is None and image_provider != 'manual':
        configured = bool(os.environ.get('GEMINI_API_URL') or os.environ.get('GEMINI_API_KEY'))
        if image_provider == 'gemini' or configured:
            from .providers.gemini import generate_normal_atlas
            request.update(status='generating', provider='gemini')
            write_json(directory / 'image_request.json', request)
            print('Generating enhanced normal atlas with Gemini API...', flush=True)
            try:
                generate_normal_atlas(reference, target, category, num_views)
            except (RuntimeError, ValueError, OSError):
                request['status'] = 'generation_failed'
                write_json(directory / 'image_request.json', request)
                raise
    deadline = time.monotonic() + wait_seconds
    if not target.is_file():
        print(f'IMAGE_REQUIRED: {directory / "image_request.json"}', flush=True)
    while not target.is_file() and time.monotonic() < deadline:
        time.sleep(2)
    if not target.is_file():
        raise ImageRequired(f'Generate the requested normal atlas and save it to {target}, then rerun to continue')
    with Image.open(target) as image:
        image.verify()
    request.update(status='image_available', generated_sha256=sha256(target))
    write_json(directory / 'image_request.json', request)
    return target


def prepare_views(generated, output_dir, num_views, resolution):
    """Split, align and mask the atlas without learned background removal.

    Black-background masks are aligned to the original rendered silhouette;
    intersection masks reject invented geometry outside that silhouette.
    """
    directory = Path(output_dir)
    columns = 2 if num_views == 4 else 3
    with Image.open(generated) as image:
        atlas = np.asarray(image.convert('RGB'))
    height, width = atlas.shape[:2]
    if abs(width / height - columns / 2) > 0.02:
        raise ValueError(f'Normal atlas must have a {columns}:2 aspect ratio, got {width}x{height}')
    paths, stats = [], []
    target_dir = directory / 'target_views'
    target_dir.mkdir(exist_ok=True)
    for index in range(num_views):
        x0, x1 = round(index % columns * width / columns), round((index % columns + 1) * width / columns)
        y0, y1 = round(index // columns * height / 2), round((index // columns + 1) * height / 2)
        rgb = cv2.resize(atlas[y0:y1, x0:x1], (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        with Image.open(directory / 'rendered_views' / f'{index:02d}.png') as frame:
            reference = np.asarray(frame.convert('RGBA').resize((resolution, resolution)))
        source_mask = (rgb.max(-1) > 30).astype(np.uint8)
        reference_mask = (reference[..., 3] > 127).astype(np.uint8)
        src, dst = cv2.moments(source_mask), cv2.moments(reference_mask)
        if not src['m00'] or not dst['m00']:
            raise ValueError(f'Empty generated/reference foreground in view {index}')
        scale = np.sqrt(dst['m00'] / src['m00'])
        if not 0.5 < scale < 2:
            raise ValueError(f'Generated view {index} differs too much in scale: {scale:.3f}')
        tx = dst['m10'] / dst['m00'] - scale * src['m10'] / src['m00']
        ty = dst['m01'] / dst['m00'] - scale * src['m01'] / src['m00']
        transform = np.float32([[scale, 0, tx], [0, scale, ty]])
        rgb = cv2.warpAffine(rgb, transform, (resolution, resolution))
        warped_mask = cv2.warpAffine(source_mask, transform, (resolution, resolution), flags=cv2.INTER_NEAREST).astype(bool)
        mask = warped_mask & reference_mask.astype(bool)
        union = warped_mask | reference_mask.astype(bool)
        overlap = float(mask.sum() / max(union.sum(), 1))
        if overlap < 0.65:
            raise ValueError(f'Generated view {index} violates reference silhouette (IoU={overlap:.3f})')
        normal = rgb.astype(np.float32) / 127.5 - 1
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
        projected = ((normal + 1) * 127.5).clip(0, 255).astype(np.uint8)
        projected[~mask] = 0
        rgba = np.concatenate([projected, mask[..., None].astype(np.uint8) * 255], -1)
        path = target_dir / f'{index:02d}.png'
        Image.fromarray(rgba).save(path)
        paths.append(path)
        stats.append({'view': index, 'silhouette_iou': overlap, 'alignment_scale': float(scale)})
    write_json(directory / 'image_alignment.json', stats)
    return paths
