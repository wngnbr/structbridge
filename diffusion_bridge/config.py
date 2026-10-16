"""One configuration source for model construction, training and inference."""

from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "default.yaml"


def load_config(path=None, overrides=(), checkpoint_config=None, *, validate=True):
    """Merge defaults, embedded checkpoint config, YAML and explicit overrides.

    Unknown keys are rejected, including obsolete options that used to have no
    effect. Return an independent config; model constructors must not mutate it.
    """
    config = OmegaConf.load(DEFAULT_CONFIG)
    OmegaConf.set_struct(config, True)
    if checkpoint_config is not None:
        config = OmegaConf.merge(config, checkpoint_config)
    if path:
        config = OmegaConf.merge(config, OmegaConf.load(path))
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    if validate:
        validate_config(config)
    return config


def validate_config(config):
    for section in (config.model, config.vae.model):
        if section.width <= 0 or section.heads <= 0 or section.width % section.heads:
            raise ValueError("Model width must be positive and divisible by the head count")
    if config.model.layers <= 0:
        raise ValueError("model.layers must be positive")
    if config.model.input_channels != config.vae.model.embed_dim or config.model.output_channels != config.vae.model.embed_dim:
        raise ValueError("Bridge input/output channels must match the VAE embedding dimension")
    if not config.vae.model.use_downsample or config.vae.model.embed_point_feats:
        raise ValueError("The supported shape VAE uses FPS queries and unembedded normals")
    if config.vae.model.tokens_per_group <= 0:
        raise ValueError("vae.model.tokens_per_group must be positive")
    if not 0 <= config.model.drop_path < 1:
        raise ValueError("model.drop_path must be in [0, 1)")
    if config.training.global_batch_size <= 0 or config.training.batch_size == 0 or config.training.batch_size < -1:
        raise ValueError("Training batch sizes must be positive (or batch_size=-1)")
    if config.training.log_interval <= 0 or config.training.save_interval < 0 or config.training.max_steps < 0:
        raise ValueError("Invalid training log/save interval or maximum step count")
    if config.training.microbatch == 0 or config.training.microbatch < -1:
        raise ValueError("training.microbatch must be positive or -1")
    if not config.training.ema_rates or not all(0 < rate < 1 for rate in config.training.ema_rates):
        raise ValueError("EMA rates must be a nonempty list of values in (0, 1)")
    if config.training.tracking_mode not in ("disabled", "offline", "online"):
        raise ValueError("tracking_mode must be disabled, offline, or online")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers must be nonnegative")
    if config.sampling.sampler not in ("dbim", "dbim_high_order", "heun"):
        raise ValueError("sampling.sampler must be dbim, dbim_high_order, or heun")
    if config.sampling.steps <= 0 or config.sampling.octree_depth <= 0:
        raise ValueError("Sampling steps and octree depth must be positive")
    if not 0 <= config.sampling.eta <= 1:
        raise ValueError("sampling.eta must be in [0, 1]")


def to_dict(config):
    return OmegaConf.to_container(config, resolve=True)


def save_config(config, path):
    OmegaConf.save(config, path)


def add_config_arguments(parser):
    parser.add_argument("--config", help="YAML file overriding the canonical configuration")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a dotted config key; may be repeated")


def apply_cli(config, args, mapping, *, validate=True):
    """Apply only explicitly provided CLI values, followed by --set overrides."""
    for name, key in mapping.items():
        value = getattr(args, name, None)
        if value is not None:
            OmegaConf.update(config, key, value)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))
    if validate:
        validate_config(config)
    return config


def str2bool(value):
    import argparse
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")
