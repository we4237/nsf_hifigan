import os
import json
import glob
import argparse
from typing import Optional
import numpy as np
import torch
from torch.nn import functional as F
import torchaudio
from nsf_hifigan.model.nsf_hifigan import NSF_HifiGAN
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from nsf_hifigan.hparams import HParams
import pytorch_lightning as pl
from cmy.melf0 import librosa_pad_lr
import librosa
import numpy as np #1.20
import parselmouth
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from nsf_hifigan.mel_processing import wav2mel
from nsf_hifigan.model.pipeline import AudioPipeline
from nsf_hifigan.data.dataset import get_pitch, load_audio
from nsf_hifigan.model.loss import  feature_loss, generator_loss

def get_hparams(config_path: str) -> HParams:
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)
    return hparams

def get_train_params(args, hparams):
    devices = [int(n.strip()) for n in args.device.split(",")]

    trainer_params = {
        "accelerator": args.accelerator,
    }

    if args.accelerator != "cpu":
        trainer_params["devices"] = devices

    if len(devices) > 1:
        trainer_params["strategy"] = "ddp"

    trainer_params.update(hparams.trainer)

    if hparams.train.fp16_run:
        trainer_params["amp_backend"] = "native"
        trainer_params["precision"] = 16
    
    trainer_params["num_nodes"] = args.num_nodes

    return trainer_params


def wav2melf0(wav_path):
    fft_size=512
    hop_size=128
    win_length=512
    window="hann"
    num_mels=80
    fmin=30
    fmax=12000
    eps=1e-6
    sample_rate=24000

    wav, _ =  librosa.core.load(wav_path, sr=sample_rate)
    wav1 = load_audio(wav_path, sr=sample_rate)

    audio_pipeline = AudioPipeline(sample_rate=sample_rate,
                                    fft_size=fft_size,
                                    num_mels=num_mels,
                                    win_length=win_length,
                                    hop_size=hop_size)
    wav_ = torch.from_numpy(wav)
    wav,mel = audio_pipeline(wav_) # (n_mel_bins, T)
    
    f0 = get_pitch(wav,mel,
                   sample_rate,
                    fft_size,
                    win_length,
                    hop_size)


    return mel,f0,wav.shape[0]

def loss(raw_wav,raw_mel,wav_hat) ->  np.ndarray:
    with torch.no_grad():
        new_mel = wav2mel(wav_hat.squeeze(0).float(),
                    512,
                    80,
                    24000,
                    128,
                    512,
                    center=False,
                    complex=False)

        # y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = model.net_period_d(raw_wav,wav_hat)
        # loss_p_fm = feature_loss(fmap_p_r, fmap_p_g)
        # loss_p_gen, losses_p_gen = generator_loss(y_dp_hat_g)

        # y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = model.net_scale_d(raw_wav, wav_hat)
        # loss_s_fm = feature_loss(fmap_s_r, fmap_s_g)
        # loss_s_gen, losses_s_gen = generator_loss(y_ds_hat_g)
    new_mel = new_mel[:,:,:raw_mel.shape[2]].cuda()
    # mel
    loss_spec = F.l1_loss(new_mel, raw_mel) * 45
        # loss_gen_all = (loss_s_gen + loss_s_fm) + (loss_p_gen + loss_p_fm) + loss_spec
    
    return loss_spec.cpu().numpy() 
    # return loss_gen_all.cpu().numpy()

def plot(mel0,mel1,mel2=None,name='0607.png'):  # 创建折线图
    if mel2.all() != None:
        dif1 = abs(mel0-mel1)
        dif2 = abs(mel0-mel2)
        dif_list_1 = [x for x in dif1]
        dif_list_2 = [x for x in dif2]
        plt.plot(dif_list_1, label='raw')
        plt.plot(dif_list_2, label='last')
        # 添加图例
        plt.legend()
        # 保存图像
        plt.savefig(name)
        plt.close()
    else:
        dif0 = abs(mel0-mel1)
        dif_list = [x.mean() for x in dif0]

        plt.plot(dif_list, label='last')
        
        # 添加图例
        plt.legend()
        # 保存图像
        plt.savefig(name)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/24k.json", help='JSON file for configuration')
    parser.add_argument('-a', '--accelerator', type=str, default="gpu", help='training device')
    parser.add_argument('-d', '--device', type=str, default="0", help='training device ids')
    parser.add_argument('-n', '--num-nodes', type=int, default=1, help='training node number')
    args = parser.parse_args()

    hparams = get_hparams(args.config)
    # lightning_fabric.utilities.seed.seed_everything(hparams.train.seed)
    pl.utilities.seed.seed_everything(hparams.train.seed)

    devices = [int(n.strip()) for n in args.device.split(",")]

    trainer_params = get_train_params(args, hparams)

    
    model = NSF_HifiGAN(**hparams).cuda()

    newtrain_ckpt = torch.load('logs_bak/version_0629_raw/checkpoints/epoch=120-step=14000.ckpt',map_location="cpu")["state_dict"]
    model.load_state_dict(newtrain_ckpt,strict=True)

    pretrain_ckpt = torch.load('cp/model_ckpt_steps_1512000.ckpt',map_location="cpu")["state_dict"]
    new_pretrain_ckpt = {}
    # 修改 model_gen 中的键
    for k, v in pretrain_ckpt['model_gen'].items():
        new_key = 'net_g.' + k
        new_pretrain_ckpt[new_key] = v
    # 修改 model_disc 中的键
    for k, v in pretrain_ckpt['model_disc'].items():
        if k.startswith('mpd.'):
            new_key = 'net_period_d.' + k[4:]
        elif k.startswith('msd.'):
            new_key = 'net_scale_d.' + k[4:]
        else:
            new_key = k
        new_pretrain_ckpt[new_key] = v
    model_raw = NSF_HifiGAN(**hparams).cuda()
    # new_pretrain_ckpt = torch.load('',map_location="cpu")["state_dict"]
    model_raw.load_state_dict(new_pretrain_ckpt,strict=True)

    gt_path = 'dataset/data_24k/001_030_24k.wav'
    raw_wav,_ = librosa.core.load(gt_path, sr=24000)

    gt_mel,gt_f0,mask = wav2melf0(gt_path)
    gt_mel = gt_mel.unsqueeze(0)
    mel = gt_mel.cuda() # (1, n_mel_bins, T)
    f0 = gt_f0.cuda()
    test_mel = gt_mel.cuda()

    npy = 'dataset/data_24k/001_030.npy'
    gt_mel = torch.from_numpy(np.load(npy,allow_pickle=True)[1]) # (T,n_mel_bins) 
    gt_mel = gt_mel.transpose(2,1) # (1,n_mel_bins,T) 
    gt_f0 = torch.from_numpy(np.load(npy,allow_pickle=True)[2])


    # e2e
    # gt_mel = torch.from_numpy(np.load('mel_e2e.npy')).transpose(1,2)
    # gt_f0 = torch.from_numpy(np.load('f0_e2e.npy'))


    
    mel = gt_mel.cuda() # (1, n_mel_bins, T)
    f0 = gt_f0.cuda()
    # mask = f0.shape[1]

    model.eval()
    with torch.no_grad():
        y_wav_raw = model_raw.net_g(mel,f0)
        y_wav = model.net_g(mel,f0)
    y_wav_result = y_wav.cpu().numpy()
    wavfile.write('audio_0629_raw.wav', 24000, y_wav_result)

    raw_wav = torch.from_numpy(raw_wav).unsqueeze(0).unsqueeze(1).cuda()
    loss_0 = loss(raw_wav=raw_wav,raw_mel=test_mel,wav_hat=y_wav_raw)
    loss_new = loss(raw_wav=raw_wav,raw_mel=test_mel,wav_hat=y_wav)

    # mel_hat_0 = wav2mel(y_wav_raw.squeeze().float(),
    #                 512,
    #                 80,
    #                 24000,
    #                 128,
    #                 512,
    #                 center=False,
    #                 complex=False)
    # mel_hat_new = wav2mel(y_wav.squeeze().float(),
    #             512,
    #             80,
    #             24000,
    #             128,
    #             512,
    #             center=False,
    #             complex=False)  
    # loss_0 =  F.l1_loss(mel_hat_0, gt_mel) * 45
    # loss_new = F.l1_loss(mel_hat_new, gt_mel) * 45


    print(f'新vocoder:{loss_new}')
    print(f'原vocoder:{loss_0}') 
