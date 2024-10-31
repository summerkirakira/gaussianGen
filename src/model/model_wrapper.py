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
from ..misc.render_utils import VideoCreator
import wandb


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
        labels = self.get_labels_batch(batch)
        skeletons = self.get_skeleton_batch(batch)
        losses, output = self.diffusion_model.training_losses(self.unet, features, t, label=labels,
                                                              skeleton_points=skeletons)

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

        if self.global_step % self.cfg.trainer.log_images_every_n_steps == 0 and self.global_step > 0:
            self.log_image(pred_images[0], original_images[0], "image")
            with torch.no_grad():
                camera_test = MiniCam.get_cam(distance=1.4, theta=3.14 / 4, phi=3.14 / 4)
                sample = inference(self.diffusion_model, self.unet, self.device, config=self.cfg.inference)
                sample = sample.permute(0, 2, 3, 4, 1).reshape(1, -1, 32)
                with torch.amp.autocast('cuda', enabled=False):
                    image = self.decoder.render(camera_test, sample[0])[0]
                self.log_single_image(image, 'Inference')
        elif self.global_step % self.cfg.trainer.log_videos_every_n_steps == 0:
            self.render_video()
        return loss

    def render_video(self):
        with torch.no_grad():
            n_images = self.cfg.inference.n_images

            theta_interval = 2 * 3.1416 / n_images
            thetas = [theta_interval * i for i in range(n_images)]

            cameras = []

            for theta in thetas:
                camera = MiniCam.get_cam(
                    self.cfg.inference.image_width,
                    self.cfg.inference.image_height,
                    self.cfg.inference.distance,
                    theta,
                    self.cfg.inference.phi
                )
                cameras.append(camera)
            white_bg = self.cfg.inference.background_color == "white"
            if not self.cfg.inference.conditional_generation:
                images = self.inference_unconditioned(cameras, white_bg)
            else:
                import clip
                model, preprocess = clip.load("ViT-B/32", device="cuda")
                text_input = clip.tokenize(self.cfg.inference.condition.label_text).cuda()
                text_features = model.encode_text(text_input).float()
                images = self.inference_conditioned(cameras, label=text_features, white_background=white_bg)

            video_path = str((Path(self.cfg.output_path) / "output_video.mp4").absolute())

            with VideoCreator(video_path, fps=self.cfg.inference.video.frame_rate) as creator:
                success = creator.create_video(
                    pil_images=images,
                    progress_bar=True
                )
                if not success:
                    print("Failed to create video")
                else:
                    wandb.log({"Rendered Video": wandb.Video(video_path, fps=self.cfg.inference.video.frame_rate, format="mp4")})

    def get_predicted_image(self, features: Tensor, camera_gts: list[MiniCam]) -> Tensor:
        pred_images = []
        for i in range(features.shape[0]):
            pred_images.append(self.decoder.render(camera_gts[i], features[i])[0])
        return torch.stack(pred_images, dim=0)

    def get_features_batch(self, batch: TrainDataGaussianType):
        feature_list = []
        for i in range(len(batch.anchor_feature)):
            feature_list.append(
                batch.anchor_feature[i].reshape(32, 32, 32, 32).permute(3, 0, 1, 2)
            )
        features = torch.stack(feature_list, dim=0)
        return features

    def get_labels_batch(self, batch: TrainDataGaussianType):
        label_list = []
        if batch.label_feature is None:
            return None
        for i in range(len(batch.label_feature)):
            label_list.append(
                batch.label_feature[i].squeeze(0)
            )
        labels = torch.stack(label_list, dim=0)
        return labels

    def get_skeleton_batch(self, batch: TrainDataGaussianType):
        skeleton_list = []
        if batch.skeleton_points is None:
            return None
        for i in range(len(batch.skeleton_points)):
            skeleton_list.append(
                batch.skeleton_points[i].squeeze(0)
            )
        skeleton_list = torch.stack(skeleton_list, dim=0)
        return skeleton_list

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
        optimizer = torch.optim.Adam(lr=self.cfg.trainer.learning_rate, params=self.model_parameters)
        return optimizer

    def inference_unconditioned(self, cameras: list[MiniCam], white_background=False) -> list[Image]:
        images: list[Image] = []
        with torch.no_grad():
            sample = inference(self.diffusion_model, self.unet, self.device, config=self.cfg.inference)
            sample = sample.permute(0, 2, 3, 4, 1).reshape(1, -1, 32)
            # xyz, color, opacity, scaling, rot, neural_opacity, mask = self.decoder.get_gaussian_properties(cameras[0], sample[0])
            for camera in cameras:
                with torch.amp.autocast('cuda', enabled=False):
                    image = self.decoder.render(camera, sample[0], white_background)[0]
                    # image, _ = self.decoder._render_gs(camera, xyz, color, opacity, scaling, rot, neural_opacity, mask)
                image = torch.clamp(image, 0, 1)

                image = image.permute(1, 2, 0).cpu().numpy()
                image = (image * 255).astype('uint8')
                image = Image.fromarray(image)
                images.append(image)
        return images

    def inference_conditioned(self, cameras: list[MiniCam], label: Optional[Tensor],
                              skeleton_points: Optional[Tensor] = None, white_background=False) -> list[Image]:
        images: list[Image] = []
        with torch.no_grad():
            sample = inference(self.diffusion_model, self.unet, self.device, label=label,
                               skeleton_points=skeleton_points, config=self.cfg.inference)
            sample = sample.permute(0, 2, 3, 4, 1).reshape(1, -1, 32)
            # xyz, color, opacity, scaling, rot, neural_opacity, mask = self.decoder.get_gaussian_properties(cameras[0], sample[0])
            for camera in cameras:
                with torch.amp.autocast('cuda', enabled=False):
                    image = self.decoder.render(camera, sample[0], white_background)[0]
                    # image, _ = self.decoder._render_gs(camera, xyz, color, opacity, scaling, rot, neural_opacity, mask)
                image = torch.clamp(image, 0, 1)

                image = image.permute(1, 2, 0).cpu().numpy()
                image = (image * 255).astype('uint8')
                image = Image.fromarray(image)
                images.append(image)
        return images
