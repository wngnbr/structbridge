"""Regression checks for normal coordinates, image handoff and cache validation."""

import json
import shutil
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh

from normal_refinement.images import ImageRequired, prepare_views, require_generated_image, write_json
from normal_refinement.optimization import gaussian_blur, make_targets, normal_loss
from normal_refinement.pipeline import create_parser, run
from normal_refinement.rendering import cameras, normalize_mesh, restore_mesh, render_normals
from normal_refinement.utils.func import load_obj
from normal_refinement.utils.render import NormalsRenderer
from normal_refinement.core.remesh import calc_vertex_normals


class NormalCoordinateTests(unittest.TestCase):
    def test_input_coordinate_frame_is_restored(self):
        vertices = torch.tensor([[1., 2., -5.], [4., 8., 9.], [-2., 3., 4.]])
        original = vertices.clone()
        normalized, center, radius = normalize_mesh(vertices)
        torch.testing.assert_close(vertices, original)
        torch.testing.assert_close(restore_mesh(normalized, center, radius), original)

    def test_six_cameras_are_rotations_and_roundtrip_normals(self):
        view, _ = cameras(6, device='cpu')
        self.assertEqual(tuple(view.shape), (6, 4, 4))
        rotations = view[:, :3, :3]
        torch.testing.assert_close(rotations @ rotations.transpose(1, 2), torch.eye(3).expand(6, -1, -1), atol=1e-6, rtol=0)
        normals = F.normalize(torch.randn(6, 7, 3), dim=-1)
        camera = torch.einsum('bij,bpj->bpi', rotations, normals)
        recovered = torch.einsum('bpj,bjk->bpk', camera, rotations)
        torch.testing.assert_close(recovered, normals)
        torch.testing.assert_close(view[4, :3, :3], view[5, :3, :3].T, atol=1e-6, rtol=0)

    def test_separable_filter_matches_original_2d_kernel(self):
        inputs = torch.randn(2, 3, 19, 17, dtype=torch.float64)
        x = torch.arange(7, dtype=torch.float64) - 3
        kernel = torch.exp(-x.square() / (2 * 1.4**2))
        kernel /= kernel.sum()
        kernel_2d = torch.outer(kernel, kernel)[None, None].expand(3, 1, 7, 7)
        expected = F.conv2d(F.pad(inputs, (3, 3, 3, 3), mode='replicate'), kernel_2d, groups=3)
        torch.testing.assert_close(gaussian_blur(inputs, 7, 1.4), expected, rtol=1e-12, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA renderer required')
    def test_six_view_render_to_target_preserves_normal_convention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / 'sphere.obj'
            trimesh.creation.icosphere(subdivisions=2).export(mesh)
            render_normals(mesh, root, resolution=64, num_views=6)
            vertices, faces = load_obj(mesh)
            vertices, _, _ = normalize_mesh(vertices)
            mv, projection = cameras(6)
            renderer = NormalsRenderer(mv, projection, (64, 64))
            reference = renderer.render(vertices, calc_vertex_normals(vertices, faces), faces)
            paths = [root / 'rendered_views' / f'{i:02d}.png' for i in range(6)]
            target = make_targets(paths, reference, mv, frequency_separation=False)
            # Rasterization interpolates vertex normals without renormalizing;
            # target preparation projects them to the unit sphere. Compare in
            # that same convention, allowing only PNG quantization error.
            canonical = reference.clone()
            canonical[..., :3] = (F.normalize(reference[..., :3] * 2 - 1, dim=-1) + 1) / 2
            _, loss, _, count = normal_loss(canonical, target, mv)
            self.assertGreater(int(count), 0)
            self.assertLess(float(loss), 1e-5)


class ImageHandoffTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA renderer required')
    def test_six_view_pipeline_exports_restored_obj_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.obj'
            mesh = trimesh.creation.icosphere(subdivisions=1, radius=2.)
            mesh.apply_translation([2., 3., 4.])
            mesh.export(source)
            output = root / 'result'
            args = create_parser().parse_args([
                '--mesh_path', str(source), '--out_dir', str(output), '--num_views', '6',
                '--render_resolution', '64', '--resolution', '64', '--steps', '2',
                '--edge_min', '0.04', '--edge_max', '0.1',
            ])
            with self.assertRaises(ImageRequired):
                run(args)
            # Identity target is a test fixture, not a production generation fallback.
            shutil.copy2(output / 'rendered_normals.png', output / 'generated_normals.png')
            result = run(args)
            refined = trimesh.load(result, force='mesh')
            self.assertTrue(np.isfinite(refined.vertices).all())
            np.testing.assert_allclose(refined.bounds.mean(0), [2., 3., 4.], atol=.05)
            self.assertFalse((output / 'final.pending.obj').exists())
            timestamp = result.stat().st_mtime_ns
            self.assertEqual(run(args), result)
            self.assertEqual(result.stat().st_mtime_ns, timestamp)

    def test_missing_image_creates_request_and_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'render.png'
            Image.new('RGB', (64, 64), 'blue').save(reference)
            with self.assertRaises(ImageRequired):
                require_generated_image(reference, root, 'chair', 4)
            request = json.loads((root / 'image_request.json').read_text())
            self.assertEqual(request['status'], 'image_required')
            self.assertFalse((root / 'generated_normals.png').exists())
            self.assertFalse((root / 'final.obj').exists())
            Image.new('RGB', (64, 64), 'blue').save(root / 'generated_normals.png')
            self.assertEqual(require_generated_image(reference, root, 'chair', 4), root / 'generated_normals.png')

    def test_masks_and_atlas_geometry_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'rendered_views').mkdir()
            frame = np.zeros((64, 64, 4), dtype=np.uint8)
            frame[10:54, 15:49] = (128, 128, 255, 255)
            for index in range(4):
                Image.fromarray(frame).save(root / 'rendered_views' / f'{index:02d}.png')
            atlas = np.tile(frame[..., :3], (2, 2, 1))
            generated = root / 'generated.png'
            Image.fromarray(atlas).save(generated)
            paths = prepare_views(generated, root, 4, 64)
            self.assertEqual(len(paths), 4)
            result = np.asarray(Image.open(paths[0]))
            self.assertTrue(np.array_equal(result[..., 3], frame[..., 3]))
            Image.new('RGB', (100, 64), 'blue').save(generated)
            with self.assertRaisesRegex(ValueError, 'aspect ratio'):
                prepare_views(generated, root, 4, 64)
            Image.new('RGB', (128, 128), 'black').save(generated)
            with self.assertRaisesRegex(ValueError, 'Empty'):
                prepare_views(generated, root, 4, 64)

    def test_changed_mesh_rejects_stale_directory_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.obj'
            source.write_text('different mesh')
            output = root / 'output'
            output.mkdir()
            write_json(output / 'manifest.json', {'input': {'mesh_sha256': 'obsolete'}})
            args = create_parser().parse_args(['--mesh_path', str(source), '--out_dir', str(output)])
            with self.assertRaisesRegex(ValueError, 'different mesh'):
                run(args)


if __name__ == '__main__':
    unittest.main()
