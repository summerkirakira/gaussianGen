from dataclasses import dataclass
from torch import Tensor


@dataclass
class GaussianRenderResult:
    render: Tensor
    viewspace_points: Tensor
    visibility_filter: Tensor
    radii: Tensor

