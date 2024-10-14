from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import Optional, Literal
from .diffusion import DiffusionConfig
from .unet import UnetConfig


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

    class Trainer(BaseModel):
        val_check_interval: int = 0
        max_steps: int = 300000
        gradient_clip_val: Optional[float] = None

    class Model(BaseModel):
        diffusion: DiffusionConfig
        unet: UnetConfig

    dataset: Dataset
    wandb: Wandb
    output_path: str
    checkpointing: Checkpointing
    trainer: Trainer = Trainer()
    model: Model

    seed: int = 114514
    mode: Literal["train", "test"] = "train"

    @classmethod
    def load_config(cls, conf: dict):
        return cls.model_validate(conf)
