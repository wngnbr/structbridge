"""Run StructBridge and normal refinement, retaining resumable per-stage outputs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from diffusion_bridge.config import apply_cli, load_config, to_dict
from diffusion_bridge.inference import CLI_MAP, create_argparser
from normal_refinement.images import sha256, write_json

ROOT = Path(__file__).resolve().parent


def run_command(command):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    # Checkpoint files are large; size and modification time identify local
    # changes without reading gigabytes on each resume. Meshes use content hashes.
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def main():
    parser = create_argparser(require_input=False)
    parser.description = __doc__
    parser.add_argument('input', nargs='?', default='mesh.glb', help='GLB/OBJ mesh or preprocessed NPZ; default: mesh.glb')
    parser.add_argument('--stage1_mesh', help='Refine an existing stage-one GLB/OBJ, skipping stage one')
    parser.add_argument('--stage1_only', action='store_true')
    parser.add_argument('--cat', help='Object category; resumes the saved category when omitted')
    parser.add_argument('--normal_image', help='Generated normal atlas for this mesh')
    parser.add_argument('--image_provider', choices=['auto', 'gemini', 'manual'], default='auto',
                        help='Generate missing normals using configured Gemini API (auto), require it (gemini), or wait for a file (manual)')
    parser.add_argument('--refine_python', default=os.environ.get('REFINE_PYTHON', os.environ.get('SHARP_PYTHON', sys.executable)))
    parser.add_argument('--sharp_python', default=os.environ.get('SHARP_PYTHON', sys.executable))
    parser.add_argument('--refine_steps', type=int, default=101)
    parser.add_argument('--refine_resolution', type=int, default=2048)
    parser.add_argument('--num_views', type=int, choices=[4, 6], default=4)
    parser.add_argument('--wait_for_image', type=float, default=0)
    parser.add_argument('--detail_strength', type=float, default=1.)
    parser.add_argument('--smooth_iter', type=int, default=0)
    args = parser.parse_args()
    source = Path(args.stage1_mesh or args.npz_path or args.input).resolve()
    if not source.is_file():
        parser.error(f'Input does not exist: {source}')
    if args.config:
        args.config = str(Path(args.config).resolve())
    config = apply_cli(load_config(args.config, validate=False), args, CLI_MAP)
    output_dir = Path(config.sampling.output_dir).resolve() if config.sampling.output_dir else ROOT / 'outputs' / source.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    config.sampling.output_dir = ''
    resolved = to_dict(config)
    settings = {name: resolved[name] for name in ('model', 'vae', 'diffusion', 'sampling')}
    identity = {'source_sha256': sha256(source), 'imported_stage1': bool(args.stage1_mesh)}
    if not args.stage1_mesh:
        identity.update(settings=settings, model=file_identity(config.sampling.model_path),
                        vae=file_identity(config.vae.checkpoint))
    state_path = output_dir / 'pipeline.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    category = args.cat or state.get('category', 'object')
    if state and state['input'] != identity:
        raise ValueError(f'{output_dir} belongs to another input or sampling configuration; choose a new --output_dir')
    state.update(input=identity, input_path=str(source), category=category)
    write_json(state_path, state)
    stage1_dir = output_dir / 'stage1'
    stage1_dir.mkdir(exist_ok=True)
    stage1_mesh = stage1_dir / 'bridge.glb'
    cached_stage1 = stage1_mesh.is_file() and state.get('stage1_sha256') == sha256(stage1_mesh)
    if not cached_stage1:
        if args.stage1_mesh:
            if source.suffix.lower() == '.glb':
                shutil.copy2(source, stage1_mesh)
            else:
                import trimesh
                trimesh.load(source, force='mesh').export(stage1_mesh)
        else:
            npz_path = source
            if source.suffix.lower() != '.npz':
                if source.suffix.lower() not in ('.glb', '.obj', '.ply'):
                    parser.error('Input must be NPZ, GLB, OBJ or PLY')
                run_command([args.sharp_python, '-c', 'import bpy; import bmesh; import fpsample'])
                prepare_dir = output_dir / 'preprocessing'
                prepare_dir.mkdir(exist_ok=True)
                reconstructed = prepare_dir / 'watertight.obj'
                npz_path = prepare_dir / 'surface.npz'
                # Failed preprocessing must not reuse a file left by an older attempt.
                reconstructed.unlink(missing_ok=True)
                npz_path.unlink(missing_ok=True)
                run_command([sys.executable, ROOT / 'watertight.py', '--input', source, '--output', reconstructed])
                if not reconstructed.is_file():
                    raise RuntimeError('Watertight reconstruction did not produce an OBJ')
                run_command([args.sharp_python, ROOT / 'sharp.py', '--input', reconstructed, '--output', npz_path])
                if not npz_path.is_file():
                    raise RuntimeError('Sharp-edge sampling did not produce an NPZ')
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
                       str(ROOT / 'sample_single.py'), '--npz_path', str(npz_path), '--output_dir', str(stage1_dir)]
            for name in CLI_MAP:
                if name == 'output_dir':
                    continue
                value = getattr(args, name)
                if value is not None:
                    command.extend([f'--{name}', str(value)])
            if args.config:
                command.extend(['--config', args.config])
            for override in args.set:
                if override.split('=', 1)[0] != 'sampling.output_dir':
                    command.extend(['--set', override])
            # Explicit --set wins over output_dir inside a user YAML.
            command.extend(['--set', f'sampling.output_dir={stage1_dir}'])
            run_command(command)
            generated = stage1_dir / f'{npz_path.stem}.glb'
            if not generated.is_file():
                raise RuntimeError(f'Stage one did not produce {generated}')
            if generated != stage1_mesh:
                generated.replace(stage1_mesh)
        state.update(status='stage1_complete', stage1_mesh=str(stage1_mesh), stage1_sha256=sha256(stage1_mesh))
        write_json(state_path, state)
    print(f'STAGE1_MESH: {stage1_mesh}', flush=True)
    if args.stage1_only:
        return
    command = [args.refine_python, '-u', '-m', 'normal_refinement.pipeline',
               '--mesh_path', str(stage1_mesh), '--out_dir', str(output_dir / 'stage2'), '--cat', category,
               '--steps', str(args.refine_steps), '--resolution', str(args.refine_resolution),
               '--num_views', str(args.num_views), '--wait_for_image', str(args.wait_for_image),
               '--image_provider', args.image_provider,
               '--detail_strength', str(args.detail_strength), '--smooth_iter', str(args.smooth_iter)]
    if args.normal_image:
        command.extend(['--normal_image', str(Path(args.normal_image).resolve())])
    state['status'] = 'stage2_running'
    write_json(state_path, state)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        state['status'] = 'image_required' if completed.returncode == 3 else 'stage2_failed'
        write_json(state_path, state)
        raise SystemExit(completed.returncode)
    result = output_dir / 'stage2' / 'final.obj'
    if not result.is_file():
        raise RuntimeError('Stage two returned without a final OBJ')
    final = output_dir / 'final.obj'
    temporary = output_dir / 'final.obj.tmp'
    shutil.copy2(result, temporary)
    temporary.replace(final)
    state.update(status='complete', final_obj=str(final), final_sha256=sha256(final))
    write_json(state_path, state)
    print(f'FINAL_MESH: {final}', flush=True)


if __name__ == '__main__':
    main()
