from lightning.pytorch import LightningModule
from typing import Optional
from pathlib import Path
from src.misc.general_utils import strip_symmetric, build_scaling_rotation,  inverse_sigmoid
import torch
from torch import nn
import numpy as np
import math


class NeuralGaussianDecoder(LightningModule):
    def __init__(self,
                 feat_dim: int = 32,
                 voxel_num: int = 32,
                 offsets: int = 10,
                 model_path: Optional[Path] = None,
                 freeze_model: bool = False,
                 bounding_box_min: np.ndarray = np.array([-0.5, -0.5, -0.5]),
                 bounding_box_max: np.ndarray = np.array([0.5, 0.5, 0.5]),
                 ):
        super().__init__()
        self.feat_dim = feat_dim
        self.voxel_num = voxel_num
        self.offsets = offsets
        self.model_path = model_path
        self.freeze_model = freeze_model
        self.bounding_box_min = bounding_box_min
        self.bounding_box_max = bounding_box_max

        self.mlp_opacity = None
        self.mlp_cov = None
        self.mlp_offsets = None
        self.mlp_color = None

        self._anchor = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.setup_functions()
        self.setup_mlp()

    def setup_mlp(self):
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim + 3, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim + 3, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7 * self.n_offsets),
        ).cuda()

        self.mlp_offsets = nn.Sequential(
            nn.Linear(self.feat_dim + 3 + self.cov_dist_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3 * self.n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim + 3, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3 * self.n_offsets),
            nn.Sigmoid()
        ).cuda()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm


        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def save_model(self, path: Path):
        model_dict = {
            'mlp_cov': self.mlp_cov.state_dict(),
            'mlp_opacity': self.mlp_opacity.state_dict(),
            'mlp_offsets': self.mlp_offsets.state_dict(),
            'mlp_color': self.mlp_color.state_dict(),
        }
        torch.save(model_dict, path)

    def load_model(self, path: Path):
        loaded_dict = torch.load(path)
        self.mlp_cov.load_state_dict(loaded_dict['mlp_cov'])
        self.mlp_opacity.load_state_dict(loaded_dict['mlp_opacity'])
        self.mlp_offsets.load_state_dict(loaded_dict['mlp_offsets'])
        self.mlp_color.load_state_dict(loaded_dict['mlp_color'])

    def save_features(self, path: Path):
        torch.save(self._anchor_feat, path)

    def load_features(self, path: Path):
        self._anchor_feat = torch.load(path)

    def freeze(self):
        self.mlp_cov.eval()
        self.mlp_opacity.eval()
        self.mlp_offsets.eval()
        self.mlp_color.eval()

        self.mlp_cov.requires_grad_(False)
        self.mlp_opacity.requires_grad_(False)
        self.mlp_offsets.requires_grad_(False)
        self.mlp_color.requires_grad_(False)


    def generate_voxel_grid(self):
        grid_scale = 0.5
        # 计算点云的边界框
        min_bound = self.bounding_box_min
        max_bound = self.bounding_box_max

        voxel_size = (max_bound - min_bound)[0] / self.voxel_num

        # 计算在每个维度上需要的体素数量
        num_voxels = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

        # 生成网格点的坐标
        x = np.arange(num_voxels[0]) * voxel_size + min_bound[0]
        y = np.arange(num_voxels[1]) * voxel_size + min_bound[1]
        z = np.arange(num_voxels[2]) * voxel_size + min_bound[2]

        # 使用 meshgrid 生成 3D 网格
        xx, yy, zz = np.meshgrid(x, y, z)

        # 将网格点转换为一个 Nx3 的数组
        voxel_grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

        return voxel_grid


    def initial_voxel_grid(self):
        points = self.generate_voxel_grid()
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        scales = torch.ones((fused_point_cloud.shape[0], 6)).float().cuda()
        scales *= math.log(1 / 32)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(False))
        # self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda").requires_grad_(False)




