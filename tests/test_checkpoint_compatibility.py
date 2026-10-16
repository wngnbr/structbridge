"""The original metadata must not make the old model package mandatory."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from diffusion_bridge.checkpoints import load_vae_state_dict


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_descriptive_vae_names_accept_original_tensor_keys(self):
        pairs = {
            "encoder.input_proj.weight": "encoder.coarse_input_proj.weight",
            "encoder.input_proj1.weight": "encoder.sharp_input_proj.weight",
            "encoder.cross_attn.weight": "encoder.coarse_cross_attn.weight",
            "encoder.cross_attn1.weight": "encoder.sharp_cross_attn.weight",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({old: torch.full((2, 3), index) for index, old in enumerate(pairs)}, path)
            actual = load_vae_state_dict(path)
            self.assertEqual(set(actual), set(pairs.values()))
            for index, new in enumerate(pairs.values()):
                self.assertTrue(torch.equal(actual[new], torch.full((2, 3), index)))
            torch.save({"encoder.input_proj.weight": torch.ones(1),
                        "encoder.coarse_input_proj.weight": torch.zeros(1)}, path)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_vae_state_dict(path)

    def test_plain_state_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            expected = torch.arange(6).reshape(2, 3)
            torch.save({"encoder.weight": expected}, path)
            actual = load_vae_state_dict(path)
            self.assertEqual(list(actual), ["encoder.weight"])
            torch.testing.assert_close(actual["encoder.weight"], expected)

    def test_original_class_metadata_without_old_package(self):
        name = "craftsman.systems.shape_autoencoder"
        modules = {key: types.ModuleType(key) for key in
                   ("craftsman", "craftsman.systems", name)}
        legacy_class = type("ShapeAutoEncoderSystem", (), {"__module__": name})
        legacy_class.Config = type("Config", (), {
            "__module__": name, "__qualname__": "ShapeAutoEncoderSystem.Config",
        })
        modules[name].ShapeAutoEncoderSystem = legacy_class
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.ckpt"
            expected = torch.ones(2, 3)
            # Reproduce the class reference stored in the released checkpoint.
            with patch.dict(sys.modules, modules):
                torch.save({
                    "metadata": legacy_class.Config,
                    "state_dict": {"shape_model.encoder.weight": expected},
                }, path)
            self.assertNotIn(name, sys.modules)
            actual = load_vae_state_dict(path)
            torch.testing.assert_close(actual["encoder.weight"], expected)
            self.assertNotIn(name, sys.modules)

    def test_rejects_non_tensor_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pt"
            torch.save({"epoch": 3}, path)
            with self.assertRaises(ValueError):
                load_vae_state_dict(path)


if __name__ == "__main__":
    unittest.main()
