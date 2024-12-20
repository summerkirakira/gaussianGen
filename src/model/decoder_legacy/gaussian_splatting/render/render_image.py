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
from src.model.decoder_legacy.gaussian_splatting.gaussian_model import MiniGaussian
from torch import Tensor
from typing import Optional
from ..utils.camera_model import MiniCam
from ..utils.typing import GaussianRenderResult


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

    bg_color = torch.Tensor([1., 1., 1.]).cuda()

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

    # return rendered_image
    return GaussianRenderResult(
        render=rendered_image,
        viewspace_points=screenspace_points,
        visibility_filter=radii > 0,
        radii=radii
    )
