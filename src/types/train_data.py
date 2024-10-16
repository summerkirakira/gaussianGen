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

    features: list[TensorType]
    gaussian_model: GaussianModel

    # labels: [TensorType]

    class Config:
        arbitrary_types_allowed = True

    def move_list(self, data_list, device):
        return [data.to(device) for data in data_list]

    def move_data(self, device):
        # 移动 gaussian_model 中的数据
        self.gaussian_model.xyz = self.move_list(self.gaussian_model.xyz, device)
        self.gaussian_model.f_dc = self.move_list(self.gaussian_model.f_dc, device)
        self.gaussian_model.f_rest = self.move_list(self.gaussian_model.f_rest, device)
        self.gaussian_model.opacity = self.move_list(self.gaussian_model.opacity, device)
        self.gaussian_model.scale = self.move_list(self.gaussian_model.scale, device)
        self.gaussian_model.rot = self.move_list(self.gaussian_model.rot, device)

        # 移动 features
        self.features = self.move_list(self.features, device)
