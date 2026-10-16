"""Load tensor weights, including the original Dora Lightning checkpoint.

The original VAE checkpoint pickles a class reference in its configuration
metadata. Remap only that reference; no legacy package needs to be installed.
As with the original loader, legacy pickle checkpoints must come from a trusted
source. New plain state dictionaries can be loaded with weights_only=True.
"""

import pickle
import types
from collections.abc import Mapping

import torch


class _LegacyShapeSystem:
    class Config:
        """Placeholder for discarded Lightning/OmegaConf metadata."""


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == (
            "craftsman.systems.shape_autoencoder", "ShapeAutoEncoderSystem"
        ):
            return _LegacyShapeSystem
        return super().find_class(module, name)


_legacy_pickle = types.ModuleType("diffusion_bridge_legacy_pickle")
_legacy_pickle.Unpickler = _LegacyUnpickler
_legacy_pickle.load = pickle.load


def load_vae_state_dict(path):
    """Accept original Lightning weights or a plain VAE state dictionary."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False, pickle_module=_legacy_pickle
        )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected a checkpoint dictionary in {path}")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError(f"Expected a state dictionary in {path}")
    prefix = "shape_model."
    if any(key.startswith(prefix) for key in state):
        state = {key[len(prefix):]: value for key, value in state.items()
                 if key.startswith(prefix)}
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"No valid VAE tensor weights found in {path}")
    renamed = {}
    prefixes = {
        "encoder.input_proj.": "encoder.coarse_input_proj.",
        "encoder.input_proj1.": "encoder.sharp_input_proj.",
        "encoder.cross_attn.": "encoder.coarse_cross_attn.",
        "encoder.cross_attn1.": "encoder.sharp_cross_attn.",
    }
    for key, value in state.items():
        for old, new in prefixes.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        if key in renamed:
            raise ValueError(f"Duplicate VAE parameter after legacy migration: {key}")
        renamed[key] = value
    return renamed


TRAINING_FORMAT = "structbridge.training"
TRAINING_VERSION = 1


def load_bridge_checkpoint(path):
    """Load plain released weights or a versioned, self-contained training state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected a bridge state dictionary in {path}")
    if checkpoint.get("format") == TRAINING_FORMAT:
        required = {"version", "model", "optimizer", "ema", "scaler", "step",
                    "samples_seen", "config", "rank_states", "world_size"}
        missing = required - checkpoint.keys()
        if missing or checkpoint["version"] != TRAINING_VERSION:
            raise ValueError(f"Unsupported/incomplete training checkpoint: {path}; missing={missing}")
        return checkpoint
    if not checkpoint or not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise ValueError(f"Unsupported bridge checkpoint: {path}")
    return checkpoint


def bridge_weights(checkpoint):
    return checkpoint["model"] if checkpoint.get("format") == TRAINING_FORMAT else checkpoint


def capture_rng_state():
    """Capture Python, NumPy, CPU torch and this rank's CUDA random state."""
    import random
    import numpy as np
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2], "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    import random
    import numpy as np
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["name"], numpy_state["keys"].numpy().astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"],
                         numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Cannot restore a CUDA training state without CUDA")
        torch.cuda.set_rng_state(state["cuda"])


def atomic_save_checkpoint(checkpoint, path):
    """Publish a complete checkpoint before updating the latest-state pointer."""
    import json
    import os
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        pointer = path.parent / "latest.json"
        pointer_tmp = pointer.with_suffix(".json.tmp")
        pointer_tmp.write_text(json.dumps({"checkpoint": path.name, "step": checkpoint["step"]}) + "\n")
        os.replace(pointer_tmp, pointer)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_resume_path(value, workdir):
    """Auto-resume uses a stable workdir; it never searches a new timestamp dir."""
    import json
    from pathlib import Path
    if value != "auto":
        return Path(value) if value else None
    if not workdir:
        raise ValueError("--resume_checkpoint auto requires --workdir")
    pointer = Path(workdir) / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"No complete training checkpoint recorded in {pointer}")
    return pointer.parent / json.loads(pointer.read_text())["checkpoint"]


def validate_resume_config(saved, current):
    """Changing architecture, data order or optimizer settings is a new run."""
    for section in ("model", "vae", "diffusion"):
        if saved[section] != current[section]:
            raise ValueError(f"Cannot change {section} during resume; use --pretrained_ckpt for a new run")
    for key in ("root", "category", "category_filter"):
        if saved["data"][key] != current["data"][key]:
            raise ValueError(f"Cannot change data.{key} during resume")
    for key in ("seed", "lr", "weight_decay", "lr_anneal_steps", "global_batch_size", "batch_size",
                "microbatch", "ema_rates", "use_fp16"):
        if saved["training"][key] != current["training"][key]:
            raise ValueError(f"Cannot change training.{key} during resume")
