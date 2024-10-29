from lightning.pytorch import LightningModule
from typing import Optional, Tuple
from pathlib import Path
from src.misc.general_utils import strip_symmetric, build_scaling_rotation,  inverse_sigmoid
import torch
from torch import nn
import numpy as np
import math
from einops import repeat
from ..decoder_legacy.gaussian_splatting.utils.camera_model import MiniCam
from plyfile import PlyData, PlyElement

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


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
        self.n_offsets = offsets
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
        self.initial_voxel_grid()

    def setup_mlp(self):
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim + 3, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).to(self.device)

        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim + 3, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7 * self.n_offsets),
        ).to(self.device)

        self.mlp_offsets = nn.Sequential(
            nn.Linear(self.feat_dim + 3, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3 * self.n_offsets),
            nn.Tanh()
        ).to(self.device)

        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim + 3, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3 * self.n_offsets),
            nn.Sigmoid()
        ).to(self.device)

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

    def load_model_legacy(self, path: Path):
        self.mlp_cov.load_state_dict(torch.load(path / 'mlp_cov.pth', weights_only=True, map_location='cpu'))
        self.mlp_opacity.load_state_dict(torch.load(path / 'mlp_opacity.pth', weights_only=True, map_location='cpu'))
        self.mlp_color.load_state_dict(torch.load(path / 'mlp_color.pth', weights_only=True, map_location='cpu'))
        # self.mlp_scaling.load_state_dict(torch.load(path / 'mlp_scaling.pth'))
        self.mlp_offsets.load_state_dict(torch.load(path / 'mlp_offsets.pth', weights_only=True, map_location='cpu'))

    def load_model(self, path: Path):
        if path.is_dir():
            return self.load_model_legacy(path)
        loaded_dict = torch.load(path, weights_only=True)
        self.mlp_cov.load_state_dict(loaded_dict['mlp_cov'])
        self.mlp_opacity.load_state_dict(loaded_dict['mlp_opacity'])
        self.mlp_offsets.load_state_dict(loaded_dict['mlp_offsets'])
        self.mlp_color.load_state_dict(loaded_dict['mlp_color'])

    def save_features(self, path: Path):
        torch.save(self._anchor_feat, path)

    def load_features(self, path: Path):
        self._anchor_feat = torch.load(path, weights_only=True)

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
        fused_point_cloud = torch.tensor(np.asarray(points)).float().to(self.device)
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().to(self.device)

        scales = torch.ones((fused_point_cloud.shape[0], 6)).float().to(self.device)
        scales *= math.log(1 / 32)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(False))
        # self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self._anchor.shape[0]), device=self.device).requires_grad_(False)


    def get_gaussian_properties(self, viewpoint_camera, features):
        feat = features
        anchor = self._anchor
        grid_scaling = self.scaling_activation(self._scaling)

        ob_view = anchor - viewpoint_camera.camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist

        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]

        neural_opacity = self.mlp_opacity(cat_local_view_wodist)
        grid_offsets = self.mlp_offsets(cat_local_view_wodist)

        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity > 0.0)
        mask = mask.view(-1)
        opacity = neural_opacity[mask]
        color = self.mlp_color(cat_local_view_wodist)
        color = color.reshape([anchor.shape[0] * self.n_offsets, 3])  # [mask]
        scale_rot = self.mlp_cov(cat_local_view_wodist)
        scale_rot = scale_rot.reshape([anchor.shape[0] * self.n_offsets, 7])
        offsets = grid_offsets.view([-1, 3])

        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
        rot = self.rotation_activation(scale_rot[:, 3:7])

        offsets = offsets * scaling_repeat[:, :3]
        xyz = repeat_anchor + offsets

        return xyz, color, opacity, scaling, rot, neural_opacity, mask

    def render(self, viewpoint_camera, features) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = self.get_gaussian_properties(viewpoint_camera, features)
        rendered_image, radii = self._render_gs(viewpoint_camera, xyz, color, opacity, scaling, rot, neural_opacity, mask)
        return rendered_image, radii


    def _render_gs(self, viewpoint_camera, xyz, color, opacity, scaling, rot, neural_opacity, mask):
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        bg_color = torch.tensor([0.0, 0.0, 0.0], device=self.device).float()

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        screenspace_points = torch.zeros_like(xyz, dtype=self._anchor.dtype, requires_grad=True, device=self.device) + 0

        rendered_image, radii = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=color,
            opacities=opacity,
            scales=scaling,
            rotations=rot,
            cov3D_precomp=None
        )

        return rendered_image, radii

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, cam: MiniCam, features: torch.Tensor, path: Path):
        xyz, color, opacity, scaling, rot, neural_opacity, mask = self.get_gaussian_properties(cam,
                                                                                               features)

        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = color.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = np.zeros((xyz.shape[0], 45))
        opacities = opacity.detach().cpu().numpy()
        scale = scaling.detach().cpu().numpy()
        rotation = rot.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
