from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import Optional, Literal
from .diffusion import DiffusionConfig
from .unet import UnetConfig
from .neural_gs import NeuralGaussianConfig
from .inference import InferenceConfig


class BaseConfig(BaseModel):

    class Dataset(BaseModel):
        name: str
        batch_size: int
        num_workers: int
        path: str
        type: str

    class Wandb(BaseModel):
        project: str
        name: str
        entity: str
        version: str
        disabled: bool = False

    class Checkpointing(BaseModel):
        every_n_train_steps: int = 10000
        save_top_k: int = 3
        load: Optional[str] = None
        path: str = "outputs/checkpoints"

    class Trainer(BaseModel):
        val_check_interval: int = 0
        max_steps: int = 300000
        gradient_clip_val: Optional[float] = None
        learning_rate: float = 1e-4

    class Model(BaseModel):
        diffusion: DiffusionConfig = DiffusionConfig()
        unet: UnetConfig = UnetConfig()
        neural_gs: NeuralGaussianConfig = NeuralGaussianConfig()

        diffusion_loss_weight: float = 1.0
        l1_loss_weight: float = 1.0
        lpips_loss_weight: float = 1.0

    dataset: Optional[Dataset] = None
    wandb: Wandb
    output_path: str = "outputs"
    checkpointing: Checkpointing
    trainer: Trainer = Trainer()
    model: Model = Model()
    load_from_checkpoint: Optional[str] = None
    inference: InferenceConfig = InferenceConfig()
    is_inference: bool = False

    seed: int = 114514
    mode: Literal["train", "test"] = "train"
    use_fp16: bool = False

    @classmethod
    def load_config(cls, conf: dict):
        return cls.model_validate(conf)
