import torch.optim
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from .diffusions.temp_config import diffusion_cfg
from .diffusions import GaussianDiffusion
from mmgen.models import build_module

from ..config import BaseConfig
from typing import Any, Dict, Optional
from torch import nn
from .decoder.triplane_decoder import TriplaneDecoder


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    diffusion_model: GaussianDiffusion
    decoder: nn.Module

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.diffusion_model: GaussianDiffusion = build_module(diffusion_cfg)
        self.decoder: TriplaneDecoder = TriplaneDecoder()
        self.cfg = cfg
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        a = 1
        ...

    def validation_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        diffusion_optimizer = torch.optim.Adam(lr=1e-4, weight_decay=0, params=self.diffusion_model.parameters())
        decoder_optimizer = torch.optim.Adam(lr=1e-4, weight_decay=0, params=self.decoder.parameters())
        return [
            {"optimizer": diffusion_optimizer},
            {"optimizer": decoder_optimizer},
        ]
