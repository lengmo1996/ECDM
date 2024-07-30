import os
from typing import Any
import numpy as np
import time
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
import lightning.pytorch as pl
from PIL import Image

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities import rank_zero_info


class ImageLogger(Callback):
    def __init__(
        self,
        train_batch_frequency,
        val_batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
    ):
        super().__init__()
        self.rescale = rescale
        self.train_batch_frequency = train_batch_frequency
        self.val_batch_frequency = val_batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.tensorboard.TensorBoardLogger: self._testtube,
        }
        self.log_steps = [
            2**n for n in range(int(np.log2(self.train_batch_frequency)) + 1)
        ]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_frequency]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step
            )

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}-{:06}_e-{:06}_gs-{:06}_b.png".format(
                k, current_epoch, global_step, batch_idx
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @rank_zero_only
    def log_test_or_predict_img_local(
        self, trainer, save_dir, split, images, img_paths
    ):
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()
            if self.clamp:
                images = torch.clamp(images, -1.0, 1.0)
        # 将图像转为列表
        images = rearrange(images, "ws b c h w -> (ws b) c h w")
        images = [image for image in images]
        img_paths = [i for idx in img_paths for i in idx]
        for _, (image, img_path) in enumerate(zip(images, img_paths)):
            if self.rescale:
                image = (image + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            image = image.transpose(0, 1).transpose(1, 2).squeeze(-1)
            image = image.numpy()
            image = (image * 255).astype(np.uint8)
            filename = img_path.split("/")[-1]
            save_path = os.path.join(root, filename)
            Image.fromarray(image).save(save_path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if split == "train":
            batch_freq = self.train_batch_frequency
        elif split == "val":
            batch_freq = self.val_batch_frequency
        else:
            batch_freq = 1
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(
                check_idx, batch_freq
            )  # batch_idx % self.batch_freq == 0
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None
            )
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def log_img_test_or_predicrt(self, trainer, pl_module, outputs, split="val"):
        # gather output images from other gpus. The gathered tensor has the shape [world_size, B, C, H, W]
        images = pl_module.all_gather(outputs["x_samples"])  # √

        # gather image paths from different subprocesses. len(img_idx)==world_size
        img_paths = [None for _ in range(trainer.world_size)]
        dist.all_gather_object(img_paths, outputs["img_id"])

        # wati sync. between different processes
        trainer.strategy.barrier()
        self.log_test_or_predict_img_local(
            trainer, pl_module.logger.save_dir, split, images, img_paths
        )
        trainer.strategy.barrier()

    def check_frequency(self, check_idx, batch_freq):
        if ((check_idx % batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not pl_module.already_sample:
            self.log_img_test_or_predicrt(trainer, pl_module, outputs, split="test")

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.log_img_test_or_predicrt(trainer, pl_module, outputs, split="predict")


class AutoEnconderKLImageLogger(ImageLogger):
    def __init__(
        self,
        train_batch_frequency,
        val_batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
    ):
        super().__init__(
            train_batch_frequency,
            val_batch_frequency,
            max_images,
            clamp,
            increase_log_steps,
            rescale,
            disabled,
            log_on_batch_idx,
            log_first_step,
            log_images_kwargs,
        )

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        images = outputs["xrec"]
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()
            if self.clamp:
                images = torch.clamp(images, -1.0, 1.0)

        img_id = outputs["img_id"]

        root = os.path.join(pl_module.logger.save_dir, "images", "predict")
        for idx, image in enumerate(images):
            if self.rescale:
                image = (image + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            image = image.transpose(0, 1).transpose(1, 2).squeeze(-1)
            image = image.numpy()
            image = (image * 255).astype(np.uint8)
            filename = img_id[idx].split("/")[-1]
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(image).save(path)


class OptimizerChangingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == pl_module.pretrain_epochs:
            two_stage_optimizer, two_stage_schedulers = pl_module.configure_optimizers()
            trainer.optimizers = two_stage_optimizer
            trainer.strategy.setup_optimizers(trainer)


class OptimizerResumeCallback(Callback):
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if checkpoint["epoch"] + 1 >= pl_module.pretrain_epochs:
            trainer.fit_loop.epoch_progress.current.completed = checkpoint["epoch"]
            two_stage_optimizer, two_stage_schedulers = pl_module.configure_optimizers()
            trainer.optimizers = two_stage_optimizer
            trainer.strategy.setup_optimizers(trainer)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = (
            torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2**20
        )
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
