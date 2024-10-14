#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from torch import Tensor
from typing import Optional


class MiniGaussian:
    def __init__(self):
        self.max_sh_degree = 3
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.active_sh_degree = 3

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        opacity = torch.sigmoid(self._opacity)
        return opacity

    @property
    def get_scaling(self):
        scales = torch.exp(self._scaling)
        return scales

    @property
    def get_rotation(self):
        rotations = torch.nn.functional.normalize(self._rotation)
        return rotations

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def load_data(self, f_dc, f_rest, opacity, scaling, rotation, xyz):
        self._xyz = xyz.contiguous()
        if f_dc is not None:
            self._features_dc = f_dc.contiguous()
            self._features_rest = f_rest.contiguous()
        self._opacity = opacity.contiguous()
        self._scaling = scaling.contiguous()
        self._rotation = rotation.contiguous()
        self.active_sh_degree = self.max_sh_degree


def render_gs_cuda(
        xyz: Tensor,
        feature_dc: Tensor,
        feature_rest: Tensor,
        opacity: Tensor,
        scaling: Tensor,
        rotation: Tensor,
        camera = None,
        color_precomp: Optional[Tensor] = None
):
    gaussian_model = MiniGaussian()
    gaussian_model.load_data(
        feature_dc,
        feature_rest,
        opacity,
        scaling,
        rotation,
        xyz
    )

    bg_color = torch.Tensor([0., 0., 0.]).cuda()

    return _render(viewpoint_camera=camera, pc=gaussian_model, bg_color=bg_color, override_color=color_precomp)


def _render(viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0,
            override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=3,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return rendered_image
    # return GaussianRenderResult(
    #     render=rendered_image,
    #     viewspace_points=screenspace_points,
    #     visibility_filter=radii > 0,
    #     radii=radii
    # )
