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


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train(config):
    config = BaseConfig.load_config(config)

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
        # plugins=[SLURMEnvironment(auto_requeue=False)],
        log_every_n_steps=1,
        precision='16-mixed',
        strategy='ddp_find_unused_parameters_true'
        # enable_validation=False
    )

    torch.manual_seed(config.seed + trainer.global_rank)
    torch.multiprocessing.set_start_method('spawn')

    data_module = DataModule(config.dataset)
    if config.load_from_checkpoint:
        model_wrapper = ModelWrapper.load_from_checkpoint(config.load_from_checkpoint, cfg=config, strict=False, map_location="cpu")
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

