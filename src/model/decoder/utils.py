import torch
import torch.nn.functional as F


def sample_from_triplane(triplane, resolution=128):
    """
    从 Triplane 中采样生成体素向量。

    参数:
    triplane (torch.Tensor): 输入的 Triplane 张量，形状为 (3, 6, 128, 128)
    resolution (int): 输出体素网格的分辨率，默认为 128

    返回:
    torch.Tensor: 采样得到的体素向量，形状为 (18, resolution, resolution, resolution)
    """
    # 确保输入的 triplane 形状正确
    assert triplane.shape == (3, 6, 128, 128), f"Expected triplane shape (3, 6, 128, 128), got {triplane.shape}"

    # 生成采样网格
    grid = generate_grid(resolution)

    # 准备用于采样的网格
    grid_xy = grid[..., [0, 1]]
    grid_xz = grid[..., [0, 2]]
    grid_yz = grid[..., [1, 2]]

    # 从三个平面采样
    sampled_xy = F.grid_sample(triplane[0].unsqueeze(0), grid_xy.unsqueeze(0), align_corners=True)
    sampled_xz = F.grid_sample(triplane[1].unsqueeze(0), grid_xz.unsqueeze(0), align_corners=True)
    sampled_yz = F.grid_sample(triplane[2].unsqueeze(0), grid_yz.unsqueeze(0), align_corners=True)

    # 组合采样结果
    sampled = torch.cat([sampled_xy, sampled_xz, sampled_yz], dim=1)

    # 调整形状以匹配目标输出
    sampled = sampled.squeeze(0).permute(0, 2, 3, 1)

    return sampled


def generate_grid(resolution):
    """
    生成一个用于采样的 3D 网格。

    参数:
    resolution (int): 网格的分辨率

    返回:
    torch.Tensor: 形状为 (resolution, resolution, resolution, 3) 的网格
    """
    coords = torch.linspace(-1, 1, resolution)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
    grid = torch.stack([x, y, z], dim=-1)
    return grid.cuda()
