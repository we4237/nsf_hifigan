
import itertools
import os
from typing import Any, Dict
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchaudio
import torchaudio.transforms as T
import pytorch_lightning as pl
import torchmetrics

from nsf_hifigan.mel_processing import wav2mel
from .discriminators.multi_scale_discriminator import MultiScaleDiscriminator
from .discriminators.multi_period_discriminator import MultiPeriodDiscriminator
from .generators.generator import Generator

from .loss import discriminator_loss, kl_loss,feature_loss, generator_loss
from .. import utils
from .commons import slice_segments, rand_slice_segments, sequence_mask, clip_grad_value_


class NSF_HifiGAN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(*[k for k in kwargs])

        self.net_g = Generator(self.hparams)
        self.net_period_d = MultiPeriodDiscriminator(periods=self.hparams.model.multi_period_discriminator_periods)
        self.net_scale_d = MultiScaleDiscriminator()

        # metrics
        self.valid_spec_loss = torchmetrics.MeanMetric()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int):
        # x_wav是原始音频文件,y_wav是padded后的
        x_wav, x_wav_lengths = batch["x_wav_values"], batch["x_wav_lengths"]
        x_mel, x_mel_lengths = batch["x_mel_values"], batch["x_mel_lengths"]
        x_pitch, _ = batch["x_pitch_values"], batch["x_pitch_lengths"]
        y_wav, y_wav_lengths = batch["y_wav_values"], batch["y_wav_lengths"]

        # x_mel (Batch, n_mel_bins, T)
        x_mel, ids_slice = rand_slice_segments(x_mel, x_mel_lengths, self.hparams.train.segment_size // self.hparams.data.hop_length)
        x_pitch = slice_segments(x_pitch.unsqueeze(1), ids_slice, self.hparams.train.segment_size // self.hparams.data.hop_length).squeeze(1) # slice       
        y_wav = slice_segments(y_wav, ids_slice * self.hparams.data.hop_length, self.hparams.train.segment_size) # slice

        y_spec = wav2mel(
            y_wav.squeeze(1).float(),
            self.hparams.data.filter_length,
            self.hparams.data.n_mel_channels,
            self.hparams.data.sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length,
            False,
            False
            )
        
        # generator forward
        y_hat = self.net_g(x_mel, x_pitch)  # x_mel (n_mel_bins, T)

        y_spec_hat = wav2mel(
            y_hat.squeeze(1).float(),
            self.hparams.data.filter_length,
            self.hparams.data.n_mel_channels,
            self.hparams.data.sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length,
            False,
            False
        )

        # 0615 diffsinger
        # y_spec = y_spec[:,:,:y_spec_hat.size(2)]
        # Discriminator
        if optimizer_idx == 0:
            # MPD
            y_dp_hat_r, y_dp_hat_g, _, _ = self.net_period_d(y_wav, y_hat.detach())
            loss_disc_p, losses_disc_p_r, losses_disc_p_g = discriminator_loss(y_dp_hat_r, y_dp_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = self.net_scale_d(y_wav, y_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_p + loss_disc_s

            # log
            lr = self.optim_g.param_groups[0]['lr']
            scalar_dict = {"train/d/loss_total": loss_disc_all, "learning_rate": lr}
            scalar_dict.update({"train/d_p_r/{}".format(i): v for i, v in enumerate(losses_disc_p_r)})
            scalar_dict.update({"train/d_p_g/{}".format(i): v for i, v in enumerate(losses_disc_p_g)})
            scalar_dict.update({"train/d_s_r/{}".format(i): v for i, v in enumerate(losses_disc_s_r)})
            scalar_dict.update({"train/d_s_g/{}".format(i): v for i, v in enumerate(losses_disc_s_g)})

            image_dict = {}
            
            tensorboard = self.logger.experiment

            utils.summarize(
                writer=tensorboard,
                global_step=self.global_step, 
                images=image_dict,
                scalars=scalar_dict)
            
            return loss_disc_all
        
        # Generator
        if optimizer_idx == 1:
            y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = self.net_period_d(y_wav, y_hat)
            loss_p_fm = feature_loss(fmap_p_r, fmap_p_g)
            loss_p_gen, losses_p_gen = generator_loss(y_dp_hat_g)

            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.net_scale_d(y_wav, y_hat)
            loss_s_fm = feature_loss(fmap_s_r, fmap_s_g)
            loss_s_gen, losses_s_gen = generator_loss(y_ds_hat_g)

            # mel
            loss_spec = F.l1_loss(y_spec_hat, y_spec) * self.hparams.train.c_spec
            loss_gen_all = (loss_s_gen + loss_s_fm) + (loss_p_gen + loss_p_fm) + loss_spec

            # Logging to TensorBoard by default
            lr = self.optim_g.param_groups[0]['lr']
            scalar_dict = {"train/g/loss_total": loss_gen_all, "learning_rate": lr}
            scalar_dict.update({
                "train/g/p_fm": loss_p_fm,
                "train/g/s_fm": loss_s_fm,
                "train/g/p_gen": loss_p_gen,
                "train/g/s_gen": loss_s_gen,
                "train/g/loss_spec": loss_spec,
            })

            scalar_dict.update({"train/g/p_gen_{}".format(i): v for i, v in enumerate(losses_p_gen)})
            scalar_dict.update({"train/g/s_gen_{}".format(i): v for i, v in enumerate(losses_s_gen)})

            image_dict = {}
            
            tensorboard = self.logger.experiment
            utils.summarize(
                writer=tensorboard,
                global_step=self.global_step, 
                images=image_dict,
                scalars=scalar_dict)
            return loss_gen_all



    def validation_step(self, batch, batch_idx):
        self.net_g.eval()
        
        x_wav, x_wav_lengths = batch["x_wav_values"], batch["x_wav_lengths"]
        x_mel, x_mel_lengths = batch["x_mel_values"], batch["x_mel_lengths"]
        x_pitch, _ = batch["x_pitch_values"], batch["x_pitch_lengths"]
        y_wav, y_wav_lengths = batch["y_wav_values"], batch["y_wav_lengths"]

        y_spec = wav2mel(y_wav.squeeze(1),
            self.hparams.data.filter_length,
            self.hparams.data.n_mel_channels,
            self.hparams.data.sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length, 
            False,
            False)

        # remove else
        y_wav_hat = self.net_g(x_mel, x_pitch)[:,:,:y_wav.size(2)]
        y_hat_lengths = torch.tensor([y_wav_hat.shape[2]], dtype=torch.long)

        y_spec_hat = wav2mel(y_wav_hat.squeeze(1),
            self.hparams.data.filter_length,
            self.hparams.data.n_mel_channels,
            self.hparams.data.sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length, 
            False,
            False)

        # 0615 diffsinger
        # y_spec_hat = y_spec_hat[:,:,:y_spec.size(2)]

        image_dict = {
            "gen/spec": utils.plot_spectrogram_to_numpy(y_spec_hat[0].cpu().numpy()),
            "gt/spec": utils.plot_spectrogram_to_numpy(y_spec[0].cpu().numpy()),
        }
        audio_dict = {
            "gen/audio": y_wav_hat[0,:,:y_hat_lengths[0]].squeeze(0).float(),
            "gt/audio": x_wav[0,:,:x_wav_lengths[0]].squeeze(0).float()
        }

        spec_mask = torch.unsqueeze(sequence_mask(x_mel_lengths.long(), y_spec.size(2)), 1).to(y_spec.dtype)

        # metrics compute
        y_spec_masked = y_spec * spec_mask
        y_spec_masked_hat = y_spec_hat * spec_mask
        valid_spec_loss_step = F.l1_loss(y_spec_masked_hat, y_spec_masked)
        self.valid_spec_loss.update(valid_spec_loss_step.item())
        self.log("valid/loss_spec_step", valid_spec_loss_step.item(), sync_dist=True)

        # logging
        tensorboard = self.logger.experiment
        utils.summarize(
            writer=tensorboard,
            global_step=self.global_step, 
            images=image_dict,
            audios=audio_dict,
            audio_sampling_rate=self.hparams.data.sampling_rate,
        )
    
    def validation_epoch_end(self, outputs) -> None:
        self.net_g.eval()
        valid_spec_loss_epoch = self.valid_spec_loss.compute()
        self.log("valid/loss_spec_epoch", valid_spec_loss_epoch.item(), sync_dist=True)
        self.valid_spec_loss.reset()

    def configure_optimizers(self):
        self.optim_g = torch.optim.AdamW(
            self.net_g.parameters(),
            self.hparams.train.generator_learning_rate,
            betas=self.hparams.train.betas,
            eps=self.hparams.train.eps)
        self.optim_d = torch.optim.AdamW(
            itertools.chain(self.net_period_d.parameters(), self.net_scale_d.parameters()),
            self.hparams.train.discriminator_learning_rate,
            betas=self.hparams.train.betas,
            eps=self.hparams.train.eps)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=self.hparams.train.lr_decay)
        self.scheduler_g.last_epoch = self.current_epoch - 1
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=self.hparams.train.lr_decay)
        self.scheduler_d.last_epoch = self.current_epoch - 1

        return [self.optim_d, self.optim_g], [self.scheduler_d, self.scheduler_g]
    
