from typing import List, Optional
from pydantic import BaseModel, Field


class UnetConfig(BaseModel):
    dims: int = 3
    image_size: int = 32
    model_channels: int = 64
    num_res_blocks: int = 3
    channel_mult: List[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    attention_resolutions: List[int] = Field(default_factory=lambda: [8, 4])
    num_heads: int = 1
    num_head_channels: int = 64
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = True
    dropout: float = 0.0
    resblock_updown: bool = True
    encoder_dim: int = 4096
    encoder_channels: Optional[int] = None
    in_channels: int = 32
    out_channels: int = 64
    activation: str = "silu"
    att_pool_heads: int = 64
    disable_self_attentions: bool = False
    unconditional_gen: bool = True
    precision: str = "32"

    class Config:
        extra = "forbid"
        protected_namespaces = ()