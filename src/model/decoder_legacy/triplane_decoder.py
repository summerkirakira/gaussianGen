from lightning.pytorch import LightningModule
from torch import nn
from torch import Tensor
from .gaussian_splatting.utils.camera_model import MiniCam
from .utils import sample_from_triplane, generate_grid
import numpy as np
import torch
# from .gaussian_splatting.gaussian_model import build_covariance_from_scaling_rotation


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
            nn.Linear(64, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, n_offsets),
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, 7 * self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
            nn.SELU(True),
            nn.Linear(feature_dim, 3 * self.n_offsets),
        ).cuda()

        self.xyz_offset = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, 3 * self.n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_appearance = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        ).cuda()

        self.base_net = nn.Sequential(
            nn.Linear(feature_dim + 3, 64)
        )

        self.base_activation = nn.SiLU()

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

        voxel_features = sample_from_triplane(triplane_code, self.voxel_num)
        voxel_features = voxel_features.reshape(self.voxel_num, self.voxel_num, self.voxel_num, 18)
        voxel_positions = generate_grid(self.voxel_num)
        w2c = np.linalg.inv(viewpoint_camera.world_view_transform.transpose(0, 1).cpu().numpy())
        viewpoint_position = w2c[:3, 3]
        viewpoint_position = Tensor(viewpoint_position).cuda().unsqueeze(0).expand(voxel_positions.shape)
        voxel_positions = voxel_positions.cuda()
        relative_positions = voxel_positions - viewpoint_position

        voxel_features = torch.cat([voxel_features, relative_positions], dim=-1)

        voxel_features = self.base_activation(self.base_net(voxel_features)).reshape(-1, 64)

        voxel_appearance = self.mlp_appearance(voxel_features).reshape(voxel_nums, 1)

        view_mask = voxel_appearance > 0.45

        voxel_positions = voxel_positions.reshape(-1, 3)

        if torch.all(~view_mask):
            raise ValueError("No voxel is visible")

        if view_mask.sum() > 15000:
            topk_apperance, topk_indices = torch.topk(voxel_appearance.squeeze(), 15000)
            voxel_features = voxel_features[topk_indices]
            voxel_positions = voxel_positions[topk_indices]
            voxel_positions = voxel_positions.repeat_interleave(self.n_offsets, dim=0)
        else:
            voxel_features = voxel_features[view_mask.squeeze()]
            voxel_positions = voxel_positions[view_mask.squeeze()]
            voxel_positions = voxel_positions.repeat_interleave(self.n_offsets, dim=0)

        voxel_nums = voxel_features.shape[0]
        opacity = self.mlp_opacity(voxel_features).reshape(voxel_nums * self.n_offsets, 1)
        color = self.mlp_color(voxel_features).reshape(voxel_nums * self.n_offsets, 3)
        scaling_rot = self.mlp_cov(voxel_features).reshape(voxel_nums * self.n_offsets, 7)

        scaling = self.scaling_activation(scaling_rot[:, :3])
        rotation = self.rotation_activation(scaling_rot[:, 3:])
        xyz_offset = self.xyz_offset(voxel_features).reshape(voxel_nums * self.n_offsets, 3)

        xyz = voxel_positions + xyz_offset * (1 / self.voxel_num)

        return xyz, color, opacity, scaling, rotation



