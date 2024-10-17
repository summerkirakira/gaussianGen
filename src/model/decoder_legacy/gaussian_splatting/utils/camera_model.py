import torch
import numpy as np
import math
import random

from torch.optim.radam import radam


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi


def rotate_vector_spherical(vec, d_theta, d_phi):
    # 转换为球坐标
    r, theta, phi = cartesian_to_spherical(*vec)

    # 应用旋转
    new_theta = theta + d_theta
    new_phi = phi + d_phi

    # 确保 phi 在 [0, pi] 范围内
    new_phi = np.clip(new_phi, 0, np.pi)

    # 转换回笛卡尔坐标
    return spherical_to_cartesian(r, new_theta, new_phi)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def create_view_matrix(camera_position):
    """
    创建一个相机到世界的变换矩阵（c2w），
    使得相机坐标系的 Z 轴 (0, 0, 1) 映射到世界坐标系的原点

    参数:
    camera_position: 相机在世界坐标系中的位置

    返回:
    c2w: 4x4的相机到世界的变换矩阵
    """

    # 计算相机的z轴（前向量）
    # 这里我们将相机的z轴指向世界坐标系的原点
    forward = normalize(-camera_position)  # 从相机位置指向原点
    temp_vec = np.random.rand(3)
    right = normalize(np.cross(forward, temp_vec))
    up = np.cross(forward, right)
    w2c = np.eye(4)
    w2c[:3, :4] = viewmatrix(forward, up, camera_position)

    c2w = np.linalg.inv(w2c)

    return c2w


class MiniCam:
    def __init__(self,
                 world_view_transform: torch.Tensor,
                 width: int = 800,
                 height: int = 800,
                 fovy: float = 0.69,
                 fovx: float = 0.69,
                 znear: float = 0.01,
                 zfar: float = 100):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

    @staticmethod
    def get_random_cam(width: int, height: int):
        theta = np.random.uniform(0, 1.5 * np.pi)
        phi = np.random.uniform(0, np.pi)
        camera_pos = spherical_to_cartesian(3, theta, phi)
        # camera_pos = np.array([0, 0, 6])
        # print(f"Camera position: {camera_pos}")
        view_matrix = create_view_matrix(camera_pos)

        view_matrix = torch.tensor(view_matrix, dtype=torch.float32).transpose(0, 1).cuda()
        return MiniCam(view_matrix, width=width, height=height)

    @staticmethod
    def get_test_cam():
        test_world_view_transform = torch.Tensor(
            [[-5.3582e-01, -1.3660e-01, 8.3321e-01, 0.0000e+00],
             [-8.4433e-01, 8.6690e-02, -5.2877e-01, 0.0000e+00],
             [4.7560e-09, -9.8683e-01, -1.6179e-01, 0.0000e+00],
             [9.8963e-09, -1.6146e-08, 4.0311e+00, 1.0000e+00]]
        ).cuda()

        return MiniCam(test_world_view_transform)