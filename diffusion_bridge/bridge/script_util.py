"""Build the bridge network and diffusion process from resolved config."""

from .karras_diffusion import (
    KarrasDenoiser, VPNoiseSchedule, VENoiseSchedule, I2SBNoiseSchedule,
    DDBMPreCond, I2SBPreCond,
)
from ..config import load_config
from ..denoiser.transformer import BridgeDenoiser


def create_model_and_diffusion(config=None):
    config = load_config() if config is None else config
    model = BridgeDenoiser(config.model)
    params = config.diffusion
    if params.noise_schedule.startswith("vp"):
        schedule = VPNoiseSchedule(beta_d=params.beta_d, beta_min=params.beta_min)
        preconditioner = DDBMPreCond(schedule, sigma_data=params.sigma_data, cov_xy=params.cov_xy)
    elif params.noise_schedule == "ve":
        schedule = VENoiseSchedule(sigma_max=params.sigma_max)
        preconditioner = DDBMPreCond(schedule, sigma_data=params.sigma_data, cov_xy=params.cov_xy)
    elif params.noise_schedule.startswith("i2sb"):
        schedule = I2SBNoiseSchedule(beta_max=params.beta_max, beta_min=params.beta_min)
        preconditioner = I2SBPreCond(schedule)
    else:
        raise ValueError(f"Unsupported noise schedule: {params.noise_schedule}")
    diffusion = KarrasDenoiser(
        noise_schedule=schedule, precond=preconditioner,
        t_max=params.sigma_max, t_min=params.sigma_min,
    )
    return model, diffusion
