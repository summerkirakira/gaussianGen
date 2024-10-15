from pydantic import BaseModel
from typing import Optional
from pathlib import Path


class NeuralGaussianConfig(BaseModel):
    feat_dim: int = 32
    voxel_num: int = 32
    offsets: int = 10
    model_path: Optional[Path] = None
    freeze_model: bool = False

    class Config:
        protected_namespaces = ()

