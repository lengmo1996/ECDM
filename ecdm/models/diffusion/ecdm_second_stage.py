"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from einops import repeat

from ecdm.models.diffusion.ecdm_first_stage import ECDMFirstStage
from ecdm.models.patchgan import PatchGAN
from torch.optim.optimizer import Optimizer


__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


def contour_extraction(imgs):
    imgs = torch.mean(imgs, dim=1)
    iimgs = []
    for img in imgs:
        img_f = torch.fft.fft2(img)
        img_fshift = torch.fft.fftshift(img_f)
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        img_fshift[crow - 15 : crow + 15, ccol - 15 : ccol + 15] = 0
        img_ishift = torch.fft.ifftshift(img_fshift)
        iimg = torch.fft.ifft2(img_ishift)
        iimg = torch.abs(iimg)
        iimgs.append(iimg.unsqueeze(0))
    return torch.cat(iimgs, 0)


class ECDMSecondStage(ECDMFirstStage):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        discriminator_config,
        first_stage_ckpt=None,
        resume_ckpt_path=None,
        G_steps=2,
        D_steps=1,
        batch_size=1,
        loss_weight_diff=1.0,
        loss_weight_consistancy=0.01,
        loss_weight_mod=1.0,
        loss_weight_edge=1.0,
        loss_weight_real=1.0,
        loss_weight_fake=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(unet_config, ckpt_path=first_stage_ckpt, *args, **kwargs)
        # use manual optimization
        self.automatic_optimization = False
        self.batch_size = batch_size
        self.G_model = self.model
        # init in the __init__
        self.D_model = PatchGAN(**discriminator_config)
        self.G_steps = G_steps
        self.D_steps = D_steps

        assert (
            first_stage_ckpt is not None or resume_ckpt_path is not None
        ), "The second stage model should not be trained from scratch"

        if resume_ckpt_path is not None:
            self.init_from_ckpt(resume_ckpt_path)

        self.modality_loss_fn = torch.nn.MSELoss()

        self.loss_weight_diff = loss_weight_diff
        self.loss_weight_consistancy = loss_weight_consistancy
        self.loss_weight_mod = loss_weight_mod
        self.loss_weight_edge = loss_weight_edge
        self.loss_weight_real = loss_weight_real
        self.loss_weight_fake = loss_weight_fake
        target_real = torch.ones((self.batch_size, 1))
        target_fake = torch.zeros((self.batch_size, 1))
        self.register_buffer("target_real", target_real)
        self.register_buffer("target_fake", target_fake)

    @torch.no_grad()
    def sample(self, x, cond=None, return_intermediates=False):
        if self.sample_method == "ddpm":
            return self.ddpm_sample(
                cond, batch_size=16, return_intermediates=return_intermediates
            )
        elif self.sample_method == "ddim":
            steps = self.sample_config["steps"]
            return self.ddim_sample(
                cond, steps=steps, return_intermediates=return_intermediates
            )
        elif self.sample_method == "dpm":
            return self.dpm_sample(
                x, cond, self.G_model, return_intermediates=return_intermediates
            )

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        G_opt, D_opt = opt
        for _ in range(self.G_steps):
            G_opt.zero_grad()
            loss_generator, loss_generator_dict = self.compute_generator_loss(
                batch, batch_idx
            )
            self.manual_backward(loss_generator)
            for name, param in self.G_model.named_parameters():
                if param.grad is None:
                    print(name)
            G_opt.step()

        for _ in range(self.D_steps):
            D_opt.zero_grad()
            loss_discriminator, loss_discriminator_dict = (
                self.compute_discriminator_loss(batch, batch_idx)
            )
            self.manual_backward(loss_discriminator)
            for name, param in self.D_model.named_parameters():
                if param.grad is None:
                    print(name)
            D_opt.step()

        self.log("loss_generator", loss_generator.mean(), prog_bar=True)
        self.log("loss_discriminator", loss_discriminator.mean(), prog_bar=True)

        self.log_dict(loss_generator_dict, logger=True, on_step=True)
        self.log_dict(loss_discriminator_dict, logger=True, on_step=True)
        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        if self.use_scheduler:
            lr = self.optimizers()[0].optimizer.param_groups[0]["lr"]
            self.log(
                "lr_G_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

    def compute_generator_loss(self, batch, batch_idx):
        loss_dict = {}

        tir_img = self.get_input(batch, "tir_img")
        tir_edge = self.get_input(batch, "tir_edge")
        vis_img = self.get_input(batch, "vis_img")
        vis_edge = self.get_input(batch, "vis_edge")

        wrapped_tir_edge = self.wrap_cond(tir_edge)
        wrapped_vis_edge = self.wrap_cond(vis_edge)

        loss_diffusion, loss_diffusion_dict = self(tir_img, cond=wrapped_tir_edge)
        tir_edge_2_tir_img, _ = self.sample(
            tir_img, cond=wrapped_tir_edge, return_intermediates=False
        )
        loss_consistancy = self.modality_loss_fn(tir_edge_2_tir_img, tir_img)
        loss_generator = (
            self.loss_weight_diff * loss_diffusion
            + self.loss_weight_consistancy * loss_consistancy
        )

        vis_edge_2_tir_img, _ = self.sample(
            tir_img, cond=wrapped_vis_edge, return_intermediates=False
        )
        loss_g_mod = self.modality_loss_fn(
            self.D_model(vis_edge_2_tir_img), self.target_real
        )

        vis_edge_2_tir_img_HF = contour_extraction(vis_edge_2_tir_img)
        vis_edge_2_tir_img_HF = vis_edge_2_tir_img_HF.unsqueeze(dim=1).expand_as(
            vis_edge
        )
        loss_edge = self.modality_loss_fn(vis_edge, vis_edge_2_tir_img_HF)

        loss_generator = (
            loss_generator
            + self.loss_weight_mod * loss_g_mod
            + self.loss_weight_edge * loss_edge
        )

        loss_dict.update(loss_diffusion_dict)

        prefix = "train" if self.training else "val"
        if f"{prefix}/loss" in loss_dict.keys():
            loss_dict.pop(f"{prefix}/loss")
        loss_dict.update({f"{prefix}/loss_diffusion": loss_diffusion.mean()})
        loss_dict.update({f"{prefix}/loss_consistancy": loss_consistancy.mean()})
        loss_dict.update({f"{prefix}/loss_g_mod": loss_g_mod.mean()})
        loss_dict.update({f"{prefix}/loss_edge": loss_edge.mean()})
        loss_dict.update({f"{prefix}/loss_generator": loss_generator.mean()})

        return loss_generator, loss_dict

    def compute_discriminator_loss(self, batch, batch_idx):
        loss_dict = {}

        tir_img = self.get_input(batch, "tir_img")
        tir_edge = self.get_input(batch, "tir_edge")
        vis_img = self.get_input(batch, "vis_img")
        vis_edge = self.get_input(batch, "vis_edge")

        wrapped_tir_edge = self.wrap_cond(tir_edge)
        wrapped_vis_edge = self.wrap_cond(vis_edge)

        vis_edge_2_tir_img, _ = self.sample(tir_img, cond=wrapped_vis_edge)
        loss_fake = self.modality_loss_fn(
            self.D_model(vis_edge_2_tir_img.detach()), self.target_fake
        )
        loss_real = self.modality_loss_fn(self.D_model(tir_img), self.target_fake)

        loss_discriminator = (
            self.loss_weight_fake * loss_fake + self.loss_weight_real * loss_real
        )

        prefix = "train" if self.training else "val"
        loss_dict.update({f"{prefix}/loss_fake": loss_fake.mean()})
        loss_dict.update({f"{prefix}/loss_real": loss_real.mean()})
        loss_dict.update({f"{prefix}/loss_discriminator": loss_discriminator.mean()})

        return loss_discriminator, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        tir_img = self.get_input(batch, "tir_img")
        vis_edge = self.get_input(batch, "vis_edge")
        wrapped_vis_edge = self.wrap_cond(vis_edge)

        _, loss_dict_no_ema = self(tir_img, cond=wrapped_vis_edge)
        with self.ema_scope():
            _, loss_dict_ema = self(tir_img, cond=wrapped_vis_edge)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}

        self.log_dict(
            loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
        self.log_dict(
            loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        optimizer_idx=0,
        gradient_clip_val=0.0,
        gradient_clip_algorithm="norm",
    ) -> None:
        # TODO: 需要检查是否被正确调用
        self.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        c = self.get_input(batch, self.cond_stage_key)
        warapped_c = self.wrap_cond(c)
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope():
                samples, denoise_row = self.sample(
                    x, warapped_c, return_intermediates=True
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log


    def configure_optimizers(self):

        lr = self.learning_rate
        G_params = list(self.G_model.parameters())
        G_opt = torch.optim.AdamW(G_params, lr=lr)
        D_params = list(self.D_model.parameters())
        D_opt = torch.optim.AdamW(D_params, lr=lr)
        opt = [G_opt, D_opt]
        if self.use_scheduler:
            scheduler = self.scheduler_config
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(G_opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return opt, scheduler
        return opt, []

    @torch.no_grad()
    def generate(self, x, cond=None):
        if cond is not None:
            cond=self.wrap_cond(cond)
        return self.sample(x, cond)