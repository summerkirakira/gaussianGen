import torch.optim
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from .diffusions import GaussianDiffusion
from mmgen.models import build_module

from ..config import BaseConfig
from typing import Any, Dict, Optional
from torch import nn
from .decoder.triplane_decoder import TriplaneDecoder
from .decoder.gaussian_splatting.utils.camera_model import MiniCam
from .decoder.gaussian_splatting.render import render_gs_cuda
from src.model.losses import l1_loss, lpips_loss, ssim
import wandb
from torch import Tensor
from .scripts.diffusion_setup import create_gaussian_diffusion
from src.misc import distribute as dist
from .diffusions.unet import UNetModel
from .diffusions.resample import UniformSampler


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    diffusion_model: GaussianDiffusion
    decoder: nn.Module

    def __init__(self, cfg: BaseConfig):
        super().__init__()

        dist.setup_dist()
        torch.cuda.set_device(dist.dev())

        self.diffusion_model: GaussianDiffusion = create_gaussian_diffusion(**cfg.model.diffusion.model_dump())

        unet = UNetModel(**cfg.model.unet.model_dump())
        unet = unet.to(dist.dev())

        print("num of params: {} M".format(sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6))

        schedule_sampler = UniformSampler(cfg.model.diffusion.steps)

        # self.decoder: TriplaneDecoder = TriplaneDecoder()
        self.cfg = cfg
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        assert len(batch["features"]) == 1, "Only one image per batch"
        code = batch["features"][0]
        xyz_gt = batch["xyz"][0]
        f_dc_gt = batch["f_dc"][0]
        f_rest_gt = batch["f_rest"][0]
        opacities_gt = batch["opacity"][0]
        scale_gt = batch["scale"][0]
        rot_gt = batch["rot"][0]
        diffusion_optimizer, decoder_optimizer = self.optimizers()
        diffusion_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Diffusion Step
        # loss_diffusion, log_vars = self.diffusion_model(code_diff_reshape(code))
        # self.record_diffusion_logs(log_vars)
        # loss_diffusion.backward()
        # diffusion_optimizer.step()

        # Decoder Step
        viewpoint_camera = MiniCam.get_random_cam()
        gt_image = self.render_gs(xyz_gt, f_dc_gt, f_rest_gt, opacities_gt, scale_gt, rot_gt, viewpoint_camera).render

        xyz, color, opacity, scaling, rotation = self.decoder(viewpoint_camera, code)
        pred_image = self.render_gs(xyz, None, None, opacity, scaling, rotation, viewpoint_camera, color).render

        ll1 = l1_loss(gt_image, pred_image)
        self.log(f"decoder_l1_loss", ll1)
        lssim = ssim(gt_image, pred_image)
        self.log(f"decoder_ssim_loss", lssim)
        decoder_loss = (1.0 - 0.1) * ll1 + 0.1 * (1.0 - lssim)
        decoder_loss.backward()
        decoder_optimizer.step()

        if self.global_step % 500 == 0:
            self.log_image(pred_image, gt_image)
        print("Global Step: ", self.global_step)

    def record_diffusion_logs(self, log_vars: Dict[str, Any]):
        self.log("loss_ddpm_mse", log_vars["loss_ddpm_mse"])

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
        diffusion_optimizer = torch.optim.Adam(lr=1e-4, weight_decay=0, params=self.diffusion_model.parameters())
        decoder_optimizer = torch.optim.Adam(lr=1e-4, weight_decay=0, params=self.decoder.parameters())
        return [
            {"optimizer": diffusion_optimizer},
            {"optimizer": decoder_optimizer},
        ]
