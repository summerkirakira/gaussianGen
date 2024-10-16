from pydantic import BaseModel

class DiffusionConfig(BaseModel):
    steps: int = 1000
    learn_sigma: bool = True
    sigma_small: bool = False
    use_kl: bool = False
    noise_schedule: str = "cosine"
    predict_type: str = "xstart"
    predict_xstart: bool = True
    rescale_timesteps: bool = True
    rescale_learned_sigmas: bool = True
    min_snr: bool = False

    class Config:
        extra = "forbid"
