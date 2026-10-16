"""Small CPU regressions for configuration and interruption-safe training."""

import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf.errors import ConfigKeyError

from datasets import StatefulBatchSampler
from diffusion_bridge.bridge import logger
from diffusion_bridge.denoiser.transformer import BridgeDenoiser
from diffusion_bridge.bridge.train_util import TrainLoop
from diffusion_bridge.checkpoints import load_bridge_checkpoint, resolve_resume_path
from diffusion_bridge.config import apply_cli, load_config, to_dict
from train import CLI_MAP, create_argparser


class ToyShapes(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, index):
        value = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16 + index / 10
        return value, value * 0.8


class ToyEncoder:
    """Exercise all three host RNGs without external VAE weights."""
    def encode(self, batch):
        latent = batch + torch.randn_like(batch) * 0.01 + np.random.rand() * 0.01 + random.random() * 0.01
        coordinates = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]]).expand(len(batch), -1, -1)
        return latent, coordinates, coordinates


class ToyDiffusion:
    """A stochastic CPU loss for testing training-state mechanics in isolation."""
    t_min, t_max = 0.01, 0.99

    def training_bridge_losses(self, model, target, timesteps, model_kwargs):
        noisy = target + torch.randn_like(target) * timesteps[:, None, None]
        prediction = model(noisy, timesteps, model_kwargs['xT'])
        mse = (prediction - target).square().mean(dim=(1, 2))
        return {'loss': mse, 'xs_mse': mse}


class TrainingStateTests(unittest.TestCase):
    def make_loop(self, directory, steps, resume=None):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        config = load_config(overrides=[
            'model.width=16', 'model.layers=1', 'model.heads=4',
            'model.input_channels=4', 'model.output_channels=4', 'vae.model.embed_dim=4',
            'training.use_fp16=false', 'training.tracking_mode=disabled',
            'training.batch_size=3', 'training.microbatch=2', 'training.lr_anneal_steps=20',
            'training.log_interval=1', f'training.max_steps={steps}',
        ])
        config.training.workdir = str(directory)
        model, diffusion = BridgeDenoiser(config.model), ToyDiffusion()
        sampler = StatefulBatchSampler(7, 3, fingerprint='toy-v1')
        loader = DataLoader(ToyShapes(), batch_sampler=sampler, generator=torch.Generator().manual_seed(42))
        logger.configure(dir=str(directory), format_strs=[])
        return TrainLoop(model, diffusion, ToyEncoder(), loader, config, directory, resume)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_interruption_restores_optimizer_ema_rng_and_next_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = self.make_loop(root / 'full', 5)
            full.run_loop()
            expected = load_bridge_checkpoint(resolve_resume_path('auto', root / 'full'))
            interrupted = self.make_loop(root / 'split', 2)
            interrupted.run_loop()
            source = resolve_resume_path('auto', root / 'split')
            renamed = source.with_name('renamed-without-step.pt')
            source.rename(renamed)
            checkpoint = load_bridge_checkpoint(renamed)
            resumed = self.make_loop(root / 'split', 5, checkpoint)
            self.assertEqual(resumed.step, 2)
            resumed.run_loop()
            actual = load_bridge_checkpoint(resolve_resume_path('auto', root / 'split'))
            for key in ('model', 'optimizer', 'ema', 'scaler', 'step', 'samples_seen', 'rank_states'):
                self.assert_nested_equal(expected[key], actual[key])
            self.assertEqual(actual['samples_seen'], 15)
            self.assertEqual(actual['rank_states'][0]['sampler']['epoch'], 1)
            changed = self.make_loop(root / 'changed', 5)
            changed.config.training.lr = 0.01
            with self.assertRaisesRegex(ValueError, 'training.lr'):
                changed.restore(checkpoint)

    def test_prefetch_does_not_commit_unconsumed_batches(self):
        sampler = StatefulBatchSampler(11, 2)
        iterator = iter(sampler)
        prefetched = [next(iterator) for _ in range(5)]
        sampler.advance()
        restored = StatefulBatchSampler(11, 2)
        restored.load_state_dict(sampler.state_dict())
        self.assertEqual(next(iter(restored)), prefetched[1])
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            StatefulBatchSampler(11, 2, fingerprint='changed').load_state_dict(sampler.state_dict())

    def test_tiny_dataset_pads_all_distributed_ranks(self):
        for rank in range(4):
            sampler = StatefulBatchSampler(1, 2, world_size=4, rank=rank)
            self.assertEqual(next(iter(sampler)), [0, 0])
            sampler.advance()
            self.assertEqual(sampler.epoch, 1)


class ConfigurationTests(unittest.TestCase):
    def test_validate_after_combining_yaml_with_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.yaml'
            path.write_text('model:\n  width: 24\n')
            args = create_argparser().parse_args(['--config', str(path), '--set', 'model.heads=4'])
            config = apply_cli(load_config(path, validate=False), args, CLI_MAP)
            self.assertEqual(config.model.width, 24)
            self.assertEqual(config.model.heads, 4)

    def test_yaml_and_only_explicit_cli_override_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.yaml'
            path.write_text('training:\n  lr: 0.002\nmodel:\n  width: 128\n')
            checkpoint = to_dict(load_config())
            checkpoint['training']['lr'] = 0.001
            checkpoint['training']['seed'] = 21
            args = create_argparser().parse_args(['--config', str(path), '--lr', '0.003',
                                                 '--set', 'training.lr=0.004'])
            config = apply_cli(load_config(path, checkpoint_config=checkpoint), args, CLI_MAP)
            self.assertEqual(config.training.lr, 0.004)
            self.assertEqual(config.training.seed, 21)
            self.assertEqual(config.model.width, 128)

    def test_unknown_and_retired_options_fail_loudly(self):
        with self.assertRaises(ConfigKeyError):
            load_config(overrides=['training.test_interval=100'])
        with contextlib.redirect_stderr(io.StringIO()):
            for option in ('--test_interval', '--fp16_scale_growth'):
                with self.assertRaises(SystemExit):
                    create_argparser().parse_args([option, '1'])

    def test_inference_fp16_is_retained(self):
        from diffusion_bridge.inference import CLI_MAP, create_argparser
        args = create_argparser().parse_args(['--npz_path', 'shape.npz', '--use_fp16', 'true'])
        self.assertTrue(apply_cli(load_config(), args, CLI_MAP).sampling.use_fp16)


if __name__ == '__main__':
    unittest.main()
