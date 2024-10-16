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
from pathlib import Path
from .inference import inference



class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    diffusion_model: GaussianDiffusion
    decoder: NeuralGaussianDecoder

    def __init__(self, cfg: BaseConfig):
        super().__init__()

        self.diffusion_model: GaussianDiffusion = create_gaussian_diffusion(**cfg.model.diffusion.model_dump())

        self.decoder = NeuralGaussianDecoder(**cfg.model.neural_gs.model_dump())
        unet = UNetModel(**cfg.model.unet.model_dump())
        self.unet = unet

        # print("num of params: {} M".format(sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6))

        self.cfg = cfg
        self.automatic_optimization = False
        self.schedule_sampler = UniformSampler(cfg.model.diffusion.steps)

        self.decoder.load_model(Path('/home/summerkirakira/Documents/Code/gaussianGen/preprocess/model_pth'))
        self.decoder.freeze()


    @property
    def model_parameters(self):
            return [self.unet.parameters(), self.decoder.parameters()]

    def training_step(self, batch: TrainDataGaussianType, batch_idx):

        unet_optimizer, decoder_optimizer = self.optimizers()
        unet_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        camera_gt = MiniCam.get_random_cam()
        original_images = self.render_original(batch.gaussian_model, camera_gt)
        t, _ = self.schedule_sampler.sample(original_images.shape[0], device=dist.dev())
        features = self.get_features_batch(batch)
        losses, output = self.diffusion_model.training_losses(self.unet, features, t)
        output_features = output['x_t'].permute(0, 2, 3, 4, 1).reshape(original_images.shape[0], -1, 32)
        pred_images = self.get_predicted_image(output_features, camera_gt)
        loss_l1 = l1_loss(pred_images, original_images)
        loss_lpips = lpips_loss(pred_images, original_images)

        losses = losses["loss"].mean()

        loss = loss_l1 + loss_lpips

        self.log("loss_l1", loss_l1)
        self.log("loss_lpips", loss_lpips)

        self.log("loss", loss)

        loss.backward()

        if self.global_step % 100 == 0:
            self.log_image(pred_images[0], original_images[0], "image")
            with torch.no_grad():
                camera_random = MiniCam.get_random_cam()
                sample = inference(self.diffusion_model, self.unet)
                sample = sample.permute(0, 2, 3, 4, 1).reshape(1, -1, 32)
                image = self.decoder.render(camera_random, sample[0])[0]
                wandb.log({"sample": [wandb.Image(image)]})

        unet_optimizer.step()
        decoder_optimizer.step()

    def get_predicted_image(self, features, camera_gt):
        pred_images = []
        for i in range(features.shape[0]):
            pred_images.append(self.decoder.render(camera_gt, features[i])[0])
        return torch.stack(pred_images, dim=0)

    def get_features_batch(self, batch: TrainDataGaussianType):
        feature_list = []
        for i in range(len(batch.features)):
            feature_list.append(
                batch.features[i].reshape(32, 32, 32, 32).permute(3, 0, 1, 2)
            )
        features = torch.stack(feature_list, dim=0)
        return features

    def record_diffusion_logs(self, log_vars: Dict[str, Any]):
        self.log("loss_ddpm_mse", log_vars["loss_ddpm_mse"])

    def render_original(self, original_gs: TrainDataGaussianType.GaussianModel, viewpoint_camera: MiniCam) -> Tensor:
        image_list = []
        for i in range(len(original_gs.xyz)):
            image_list.append(
                self.render_gs(
                    original_gs.xyz[i],
                    original_gs.f_dc[i],
                    original_gs.f_rest[i],
                    original_gs.opacity[i],
                    original_gs.scale[i],
                    original_gs.rot[i],
                    viewpoint_camera
                ).render
            )
        return torch.stack(image_list, dim=0)

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
        diffusion_optimizer = torch.optim.AdamW(lr=1e-4, weight_decay=1e-5, params=self.unet.parameters())
        decoder_optimizer = torch.optim.AdamW(lr=1e-4, weight_decay=1e-5, params=self.decoder.parameters())
        return [
            {"optimizer": diffusion_optimizer},
            {"optimizer": decoder_optimizer},
        ]
