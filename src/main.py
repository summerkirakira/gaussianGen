from gc import callbacks
from pathlib import Path

import hydra
import torch
from .config import BaseConfig
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from .misc.local_logger import LocalLogger
from .misc.wandb_tools import update_checkpoint_path
from .dataset.data_module import DataModule
from .model.model_wrapper import ModelWrapper
from .model.decoder_legacy.gaussian_splatting.utils.camera_model import MiniCam
from .misc.render_utils import VideoCreator
from PIL import Image


def inference(config: BaseConfig):
    if config.load_from_checkpoint:
        model_wrapper = ModelWrapper.load_from_checkpoint(config.load_from_checkpoint, cfg=config, strict=False).cuda()
    else:
        raise ValueError("load_from_checkpoint is required for inference mode.")
    model_wrapper.eval()
    model_wrapper.freeze()

    n_images = config.inference.n_images

    theta_interval = 2 * 3.1416 / n_images
    thetas = [theta_interval * i for i in range(n_images)]

    cameras = []

    for theta in thetas:
        camera = MiniCam.get_cam(
            config.inference.image_width,
            config.inference.image_height,
            config.inference.distance,
            theta,
            config.inference.phi
        )
        cameras.append(camera)

    white_bg = config.inference.background_color == "white"

    for i in range(config.inference.video.n_videos):
        if not config.inference.conditional_generation:
            images = model_wrapper.inference_unconditioned(cameras, white_bg)
        else:
            import clip
            model, preprocess = clip.load("ViT-B/32", device="cuda")

            if config.inference.condition.input_image_path is not None:
                image = Image.open(config.inference.condition.input_image_path)
                image = preprocess(image).unsqueeze(0).to("cuda")
                query = model.encode_image(image).float()
            else:
                text_query = clip.tokenize(config.inference.condition.label_text).cuda()
                query = model.encode_text(text_query).float()
            images = model_wrapper.inference_conditioned(cameras, label=query, white_background=white_bg)

        with VideoCreator(f'test_output/test_output_{i}.mp4', fps=config.inference.video.frame_rate) as creator:
            success = creator.create_video(
                pil_images=images,
                progress_bar=True
            )


import os

CONFIG_NAME = os.getenv('CONFIG_NAME', 'local_4090')


@hydra.main(config_path="../config", config_name=CONFIG_NAME, version_base=None)
def train(config):
    config = BaseConfig.load_config(config)

    if config.is_inference:
        return inference(config)

    output_path = Path(config.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    if not config.wandb.disabled:
        logger = WandbLogger(
            project=config.wandb.project,
            name=f"{config.wandb.name}",
            # version=config.wandb.version,
            config=config.model_dump()
        )
    else:
        logger = LocalLogger()

    custom_callbacks = [ModelCheckpoint(
        dirpath=Path(config.checkpointing.path),
        every_n_train_steps=config.checkpointing.every_n_train_steps,
        save_top_k=config.checkpointing.save_top_k,
        save_last=True,
        monitor='loss',
        mode='min',
    )]

    checkpoint_path = update_checkpoint_path(config.checkpointing.load, config.wandb)

    trainer = Trainer(
        max_epochs=None,
        accelerator='gpu',
        logger=logger,
        devices='auto',
        callbacks=custom_callbacks,
        # val_check_interval=None,
        check_val_every_n_epoch=1000,
        enable_progress_bar=True,
        gradient_clip_val=config.trainer.gradient_clip_val,
        max_steps=config.trainer.max_steps,
        log_every_n_steps=1,
        precision='16-mixed',
        strategy='ddp_find_unused_parameters_true'
        # enable_validation=False
    )

    torch.manual_seed(config.seed + trainer.global_rank)
    torch.multiprocessing.set_start_method('spawn')

    data_module = DataModule(config.dataset)
    if config.load_from_checkpoint:
        model_wrapper = ModelWrapper.load_from_checkpoint(config.load_from_checkpoint, cfg=config, strict=False,
                                                          map_location="cpu")
    else:
        model_wrapper = ModelWrapper(
            config
        )
    if config.mode == "train":
        trainer.fit(model_wrapper, data_module, ckpt_path=checkpoint_path)
    else:
        raise NotImplementedError("Test mode is not implemented yet.")


if __name__ == "__main__":
    train()
