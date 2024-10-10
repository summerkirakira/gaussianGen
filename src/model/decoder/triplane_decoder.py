from lightning.pytorch import LightningModule
from torch import nn
from torch import Tensor
from .gaussian_splatting.utils.camera_model import MiniCam
from .utils import sample_from_triplane, generate_grid
import numpy as np
import torch
from .gaussian_splatting.gaussian_model import build_covariance_from_scaling_rotation


class TriplaneDecoder(nn.Module):
    def __init__(self,
                 feature_dim: int = 18,
                 n_offsets: int = 8,
                 voxel_num: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_offsets = n_offsets
        self.voxel_num = voxel_num

        self.mlp_opacity = nn.Sequential(
            nn.Linear(feature_dim + 3, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(feature_dim + 3, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, 7 * self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(feature_dim + 3, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, 3 * self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.xyz_offset = nn.Sequential(
            nn.Linear(feature_dim + 3, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, 3 * self.n_offsets),
        )

    @staticmethod
    def scaling_activation(x):
        return torch.exp(x)

    @staticmethod
    def rotation_activation(x):
        return torch.nn.functional.normalize(x)

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()

    def forward(self, viewpoint_camera: MiniCam, triplane_code: Tensor):

        voxel_nums = self.voxel_num ** 3

        voxel_features = sample_from_triplane(viewpoint_camera, triplane_code)
        voxel_features = voxel_features.reshape(-1, self.voxel_num * self.voxel_num * self.voxel_num).transpose(0, 1)
        voxel_positions = generate_grid(self.voxel_num).reshape(self.voxel_num ** 3, 3)
        w2c = np.linalg.inv(viewpoint_camera.world_view_transform)
        viewpoint_position = w2c[:3, 3]
        viewpoint_position = Tensor(viewpoint_position).cuda().unsqueeze(0).expand(voxel_positions.shape[0], -1)
        voxel_positions = voxel_positions.cuda()
        relative_positions = voxel_positions - viewpoint_position

        voxel_features = torch.cat([voxel_features, relative_positions], dim=1)
        opacity = self.mlp_opacity(voxel_features).reshape(voxel_nums * self.n_offsets, 1)
        color = self.mlp_color(voxel_features).reshape(voxel_nums * self.n_offsets, 3)
        scaling_rot = self.mlp_cov(voxel_features).reshape(voxel_nums * self.n_offsets, 7)

        scaling = self.scaling_activation(scaling_rot[:, :3])
        rotation = self.rotation_activation(scaling_rot[:, 3:])
        xyz_offset = self.xyz_offset(voxel_features).reshape(voxel_nums * self.n_offsets, 3)

        xyz = voxel_positions + xyz_offset

        return xyz, color, opacity, scaling, rotation



