from src.misc.general_utils import strip_symmetric, build_scaling_rotation
import torch


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


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


