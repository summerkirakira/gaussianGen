import random

import torch.optim
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from .diffusions import GaussianDiffusion, ModelMeanType
from ..config import BaseConfig
from typing import Any, Dict, Optional
from torch import nn
from .decoder_legacy.gaussian_splatting.render import render_gs_cuda
from src.model.losses import l1_loss, lpips_loss, ssim
import wandb
from torch import Tensor
from .scripts.diffusion_setup import create_gaussian_diffusion
from .diffusions.unet import UNetModel
from .diffusions.resample import UniformSampler
from src.types import TrainDataGaussianType
from .decoder_legacy.gaussian_splatting.utils.camera_model import MiniCam
from .decoder.neural_gaussian_decoder import NeuralGaussianDecoder
from PIL import Image
from pathlib import Path
from .inference import inference
from lpips import LPIPS
from pytorch_lightning.utilities import rank_zero_only


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

        self.cfg = cfg
        # self.automatic_optimization = False
        self.schedule_sampler = UniformSampler(cfg.model.diffusion.steps)

        # self.decoder.load_model(Path('decoder_model'))
        # self.decoder.freeze()
        self.lpips = LPIPS(net='alex').to(self.device)

    def on_save_checkpoint(self, checkpoint):
        checkpoint.pop('lpips', None)

    def lpips_loss(self, img1, img2):
        return self.lpips(img1, img2).mean()

    @property
    def model_parameters(self):
        return list(self.unet.parameters()) + list(self.decoder.parameters())

    def get_pred_x0(self, output, t):
        if self.diffusion_model.model_mean_type == ModelMeanType.START_X:
            pred_x0 = output['model_output']
        elif self.diffusion_model.model_mean_type == ModelMeanType.V:
            pred_x0 = self.diffusion_model._predict_start_from_z_and_v(x_t=output['x_t'], t=t, v=output['model_output'])
        else:
            pred_x0 = self.diffusion_model._predict_xstart_from_eps(output['x_t'], t, output['model_output'])
        return pred_x0

    def training_step(self, batch: TrainDataGaussianType, batch_idx):
        batch.move_data(self.device)

        image_size = random.choices([400, 600, 800, 1000], weights=[0.4, 0.3, 0.2, 0.1])[0]

        camera_gts = [MiniCam.get_random_cam(image_size, image_size) for _ in range(len(batch.gaussian_model.xyz))]
        with torch.amp.autocast('cuda', enabled=False):
            original_images = self.render_original(batch.gaussian_model, camera_gts)
        t, _ = self.schedule_sampler.sample(original_images.shape[0], device=self.device)
        features = self.get_features_batch(batch)
        losses, output = self.diffusion_model.training_losses(self.unet, features, t)

        pred_x0 = self.get_pred_x0(output, t)

        output_features = pred_x0.permute(0, 2, 3, 4, 1).reshape(original_images.shape[0], -1, 32)
        with torch.amp.autocast('cuda', enabled=False):
            pred_images = self.get_predicted_image(output_features, camera_gts)
        loss_l1 = l1_loss(pred_images, original_images)
        loss_lpips = self.lpips_loss(pred_images, original_images)

        diff_loss = losses["loss"].mean()

        loss = loss_l1 * self.cfg.model.l1_loss_weight + loss_lpips * self.cfg.model.lpips_loss_weight + diff_loss * self.cfg.model.diffusion_loss_weight

        self.log("loss_ddpm", diff_loss)
        self.log("loss_l1", loss_l1)
        self.log("loss_lpips", loss_lpips)
        self.log("loss", loss)

        # self.decoder.save_ply(camera_gts[0], output_features[0], Path(f"outputs/{self.global_step}.ply"))
        # raise ValueError("Stop here")

        if self.global_step % 100 == 0:
            self.log_image(pred_images[0], original_images[0], "image")
            with torch.no_grad():
                camera_random = MiniCam.get_test_cam()
                sample = inference(self.diffusion_model, self.unet, self.device)
                sample = sample.permute(0, 2, 3, 4, 1).reshape(1, -1, 32)
                with torch.amp.autocast('cuda', enabled=False):
                    image = self.decoder.render(camera_random, sample[0])[0]
                self.log_single_image(image, 'Inference')

        return loss

    def get_predicted_image(self, features: Tensor, camera_gts: list[MiniCam]) -> Tensor:
        pred_images = []
        for i in range(features.shape[0]):
            pred_images.append(self.decoder.render(camera_gts[i], features[i])[0])
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

    def render_original(self, original_gs: TrainDataGaussianType.GaussianModel,
                        viewpoint_cameras: list[MiniCam]) -> Tensor:
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
                    viewpoint_cameras[i]
                ).render
            )
        return torch.stack(image_list, dim=0)

    def render_gs(self, xyz_gt, f_dc_gt, f_rest_gt, opacities_gt, scale_gt, rot_gt, viewpoint_camera,
                  color_precomputed=None):
        return render_gs_cuda(xyz_gt, f_dc_gt, f_rest_gt, opacities_gt, scale_gt, rot_gt, viewpoint_camera,
                              color_precomputed)

    @rank_zero_only
    def log_single_image(self, image: Tensor, name: str = "image"):
        wandb.log({name: [wandb.Image(image)]})

    @rank_zero_only
    def log_image(self, image: Tensor, gt_image: Tensor, name: str = "image"):
        concatenated_img = torch.cat([image, gt_image], dim=2)
        concatenated_img *= 255
        concatenated_img[concatenated_img > 255] = 255
        wandb.log({name: [wandb.Image(concatenated_img)]})

    def validation_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        # diffusion_optimizer = torch.optim.AdamW(lr=1e-4, weight_decay=1e-5, params=self.unet.parameters())
        # decoder_optimizer = torch.optim.AdamW(lr=1e-4, weight_decay=1e-5, params=self.decoder.parameters())
        # return [
        #     {"optimizer": diffusion_optimizer},
        #     {"optimizer": decoder_optimizer},
        # ]
        # optimizer = torch.optim.AdamW(lr=1e-4, weight_decay=1e-5, params=self.model_parameters)
        optimizer = torch.optim.Adam(lr=1e-4, params=self.model_parameters)
        return optimizer
