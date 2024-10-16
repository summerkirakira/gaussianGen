from pydantic import BaseModel
from typing import Literal
from .tensor_type import TensorType


class TrainDataGaussianType(BaseModel):
    class GaussianModel(BaseModel):
        xyz: list[TensorType]
        f_dc: list[TensorType]
        f_rest: list[TensorType]
        opacity: list[TensorType]
        scale: list[TensorType]
        rot: list[TensorType]
        name: list[str]

        class Config:
            arbitrary_types_allowed = True

    features: [TensorType]
    gaussian_model: GaussianModel
    # labels: [TensorType]

    class Config:
        arbitrary_types_allowed = True