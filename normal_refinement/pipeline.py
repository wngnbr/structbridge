"""Prepare, generate normal targets, and refine a mesh with resumable artifacts."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .images import ImageRequired, prepare_views, require_generated_image, sha256, write_json

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_VERSION = 1


def run(args):
    source = Path(args.mesh_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    directory = Path(args.out_dir).resolve() if args.out_dir else source.parent / f'{source.stem}_refinement'
    directory.mkdir(parents=True, exist_ok=True)
    key = {'version': PIPELINE_VERSION, 'mesh_sha256': sha256(source), 'num_views': args.num_views,
           'render_resolution': args.render_resolution, 'category': args.cat}
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['input'] != key:
            raise ValueError('This output directory belongs to different mesh/view settings; choose a new --out_dir')
    else:
        manifest = {'input': key, 'source_mesh': str(source), 'status': 'preparing'}
        write_json(manifest_path, manifest)
    clean_mesh = directory / 'input_clean.obj'
    if not clean_mesh.exists():
        from .mesh import preprocess_mesh
        _, stats = preprocess_mesh(source, clean_mesh, target_vertices=10000)
        write_json(directory / 'preprocessing.json', stats)
    reference = directory / 'rendered_normals.png'
    if not reference.exists():
        # A separate process releases the renderer before optimization starts.
        subprocess.run([sys.executable, '-m', 'normal_refinement.render_cli',
                        '--mesh_path', str(clean_mesh), '--out_dir', str(directory),
                        '--resolution', str(args.render_resolution), '--num_views', str(args.num_views)],
                       cwd=PACKAGE_ROOT, check=True)
    manifest['status'] = 'image_required'
    write_json(manifest_path, manifest)
    try:
        generated = require_generated_image(reference, directory, args.cat, args.num_views,
                                            args.normal_image, args.wait_for_image,
                                            image_provider=args.image_provider)
    except ImageRequired:
        raise
    except (RuntimeError, ValueError, OSError):
        manifest['status'] = 'image_failed'
        write_json(manifest_path, manifest)
        raise
    image_hash = sha256(generated)
    settings = {key: getattr(args, key) for key in (
        'steps', 'resolution', 'frequency_separation', 'detail_strength', 'edge_min', 'edge_max',
        'laplacian_weight', 'silhouette_weight', 'smooth_iter')}
    final_path = directory / 'final.obj'
    if (manifest.get('completed_image_sha256') == image_hash and manifest.get('settings') == settings
            and final_path.is_file() and not args.re_remesh
            and manifest.get('final_sha256') == sha256(final_path)):
        manifest['status'] = 'complete'
        write_json(manifest_path, manifest)
        print(f'FINAL_MESH: {final_path}', flush=True)
        return final_path
    manifest.update(status='optimizing', settings=settings, generated_image=str(generated))
    write_json(manifest_path, manifest)
    paths = prepare_views(generated, directory, args.num_views, args.resolution)
    refined_path = directory / 'refined.obj'
    command = [sys.executable, '-m', 'normal_refinement.optimization', '--mesh_path', str(clean_mesh),
               '--img_paths', *map(str, paths), '--output_mesh_path', str(refined_path),
               '--out_dir', str(directory / 'optimization'), '--num_views', str(args.num_views)]
    for key, value in settings.items():
        if key == 'smooth_iter':
            continue
        if key == 'frequency_separation':
            command.append('--frequency_separation' if value else '--no-frequency_separation')
        else:
            command.extend([f'--{key}', str(value)])
    subprocess.run(command, cwd=PACKAGE_ROOT, check=True)
    pending_path = directory / 'final.pending.obj'
    pending_path.unlink(missing_ok=True)
    if args.smooth_iter:
        from .postprocess import smooth_taubin
        smooth_taubin(str(refined_path), str(pending_path), args.smooth_iter)
    else:
        shutil.copy2(refined_path, pending_path)
    import numpy as np
    import trimesh
    final_mesh = trimesh.load(pending_path, force='mesh', process=False)
    if not len(final_mesh.faces) or not np.isfinite(final_mesh.vertices).all():
        raise ValueError('Final OBJ failed the geometry check')
    pending_path.replace(final_path)
    manifest.update(status='complete', completed_image_sha256=image_hash, final_obj=str(final_path),
                    final_sha256=sha256(final_path), vertices=len(final_mesh.vertices), faces=len(final_mesh.faces))
    write_json(manifest_path, manifest)
    print(f'FINAL_MESH: {final_path}', flush=True)
    return final_path


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh_path', required=True)
    parser.add_argument('--out_dir')
    parser.add_argument('--cat', default='object')
    parser.add_argument('--normal_image', help='Generated normal atlas; otherwise uses out_dir/generated_normals.png')
    parser.add_argument('--image_provider', choices=['auto', 'gemini', 'manual'], default='auto',
                        help='auto: use configured Gemini API for missing images; manual: external file handoff')
    parser.add_argument('--wait_for_image', type=float, default=0, help='Seconds to wait for an external generator before exiting with status 3')
    parser.add_argument('--num_views', type=int, choices=[4, 6], default=4)
    parser.add_argument('--render_resolution', type=int, default=1024)
    parser.add_argument('--resolution', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=101)
    parser.add_argument('--frequency_separation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--detail_strength', type=float, default=1.)
    parser.add_argument('--edge_min', type=float, default=.0005)
    parser.add_argument('--edge_max', type=float, default=.002)
    parser.add_argument('--laplacian_weight', type=float, default=.01)
    parser.add_argument('--silhouette_weight', type=float, default=0.)
    parser.add_argument('--smooth_iter', type=int, default=0)
    parser.add_argument('--re_remesh', action='store_true', help='Rerun geometry optimization using the existing image')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.steps < 1 or min(args.resolution, args.render_resolution) < 64 or args.wait_for_image < 0 or args.smooth_iter < 0:
        parser.error('Invalid iteration count, resolution or wait time')
    if not 0 < args.edge_min <= args.edge_max:
        parser.error('Require 0 < edge_min <= edge_max')
    try:
        run(args)
    except ImageRequired as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(3)


if __name__ == '__main__':
    main()
