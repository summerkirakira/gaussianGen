from .diffusions.dpmsolver import model_wrapper as mw
from .diffusions.dpmsolver import NoiseScheduleVP, DPM_Solver
import torch
from src.model.diffusions.unet import UNetModel
from src.model.diffusions.gaussian_diffusion import GaussianDiffusion


def inference(diffusion: GaussianDiffusion, model: UNetModel, device, label=None, skeleton_points=None):
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(diffusion.betas).to(device))

    model_fn = mw(
        model,
        noise_schedule,
        model_type='x_start',
        model_kwargs={},
        guidance_type='uncond',
        guidance_scale=1.0
    )
    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type='dpmsolver++')

    sample_shape = (1, 32, 32, 32, 32)

    with torch.no_grad():
        noise = torch.randn(sample_shape, device=device) * 1.0

        samples = dpm_solver.sample(
            x=noise,
            steps=1000,
            t_start=1.0,
            t_end=1e-3,
            order=3,
            skip_type='time_uniform',
            method='multistep',
        )

        return samples