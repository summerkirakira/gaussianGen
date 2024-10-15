import torch.optim
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from .diffusions import GaussianDiffusion
from ..config import BaseConfig
from typing import Any, Dict, Optional
from torch import nn
from .decoder_legacy.gaussian_splatting.render import render_gs_cuda
from src.model.losses import l1_loss, lpips_loss, ssim
import wandb
from torch import Tensor
from .scripts.diffusion_setup import create_gaussian_diffusion
from src.misc import distribute as dist
from .diffusions.unet import UNetModel
from .diffusions.resample import UniformSampler
from src.types import TrainDataGaussianType
from .decoder_legacy.gaussian_splatting.utils.camera_model import MiniCam
from .decoder.neural_gaussian_decoder import NeuralGaussianDecoder
from PIL import Image


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    diffusion_model: GaussianDiffusion
    decoder: NeuralGaussianDecoder

    def __init__(self, cfg: BaseConfig):
        super().__init__()

        # dist.setup_dist()
        torch.cuda.set_device(dist.dev())

        self.diffusion_model: GaussianDiffusion = create_gaussian_diffusion(**cfg.model.diffusion.model_dump())

        self.decoder = NeuralGaussianDecoder(**cfg.model.neural_gs.model_dump())
        unet = UNetModel(**cfg.model.unet.model_dump())
        self.unet = unet.to(dist.dev())

        print("num of params: {} M".format(sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6))

        # self.decoder: TriplaneDecoder = TriplaneDecoder()
        self.cfg = cfg
        self.automatic_optimization = False
        self.schedule_sampler = UniformSampler(cfg.model.diffusion.steps)


    @property
    def model_parameters(self):
            return [self.unet.parameters(), self.decoder.parameters()]

    def training_step(self, batch: TrainDataGaussianType, batch_idx):
        camera_gt = MiniCam.get_random_cam()
        original_image = self.render_original(batch.gaussian_model, camera_gt).render
        



    def record_diffusion_logs(self, log_vars: Dict[str, Any]):
        self.log("loss_ddpm_mse", log_vars["loss_ddpm_mse"])

    def render_original(self, original_gs: TrainDataGaussianType.GaussianModel, viewpoint_camera: MiniCam):
        return render_gs_cuda(
            original_gs.xyz[0],
            original_gs.f_dc[0],
            original_gs.f_rest[0],
            original_gs.opacity[0],
            original_gs.scale[0],
            original_gs.rot[0],
            viewpoint_camera,
        )

    def render_gs(self, xyz_gt, f_dc_gt, f_rest_gt, opacities_gt, scale_gt, rot_gt, viewpoint_camera, color_precomputed=None):
        return render_gs_cuda(xyz_gt, f_dc_gt, f_rest_gt, opacities_gt, scale_gt, rot_gt, viewpoint_camera, color_precomputed)

    def log_image(self, image: Tensor, gt_image: Tensor, name: str = "image"):
        concatenated_img = torch.cat([image, gt_image], dim=2)
        concatenated_img *= 255
        concatenated_img[ concatenated_img > 255 ] = 255
        wandb.log({name: [wandb.Image(concatenated_img)]})

    def validation_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        diffusion_optimizer = torch.optim.AdamW(lr=1e-4, weight_decay=0, params=self.unet.parameters())
        decoder_optimizer = torch.optim.Adam(lr=1e-4, weight_decay=0, params=self.decoder.parameters())
        return [
            {"optimizer": diffusion_optimizer},
            {"optimizer": decoder_optimizer},
        ]
