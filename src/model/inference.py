from .diffusions.dpmsolver import model_wrapper as mw
from .diffusions.dpmsolver import NoiseScheduleVP, DPM_Solver
import torch
from src.model.diffusions.unet import UNetModel
from src.model.diffusions.gaussian_diffusion import GaussianDiffusion
from src.config.inference import InferenceConfig


def inference(diffusion: GaussianDiffusion, model: UNetModel, device, label=None, skeleton_points=None, config: InferenceConfig = InferenceConfig()):
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(diffusion.betas).to(device))

    model_fn = mw(
        model,
        noise_schedule,
        model_type='x_start',
        model_kwargs={},
        guidance_type='uncond' if (label is None and skeleton_points is None) else 'cond',
        guidance_scale=1.0,
        label=label,
        skeleton_points=skeleton_points,
    )
    dpm_solver = DPM_Solver(model_fn, noise_schedule=noise_schedule, algorithm_type='dpmsolver++')

    sample_shape = (1, 32, 32, 32, 32)

    with torch.no_grad():
        noise = torch.randn(sample_shape, device=device) * 1.0

        samples = dpm_solver.sample(
            x=noise,
            steps=config.inference_steps,
            t_start=1.0,
            t_end=1e-3,
            order=3,
            skip_type='time_uniform',
            method='multistep',
        )

        return samples