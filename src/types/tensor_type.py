import torch
from typing import Any
from typing_extensions import Annotated
from pydantic import BaseModel, Field, BeforeValidator


def validate_tensor(v):
    if isinstance(v, torch.Tensor):
        return v
    elif isinstance(v, (list, tuple)):
        return torch.tensor(v)
    elif isinstance(v, (int, float)):
        return torch.tensor([v])
    raise ValueError(f'Cannot convert {type(v)} to torch.Tensor')

TensorType = Annotated[torch.Tensor, BeforeValidator(validate_tensor)]

