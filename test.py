import os
import json
import glob
import argparse
from typing import Optional
import numpy as np
import torch
import torchaudio
import tqdm
from torch import nn, optim
from torch.nn import functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from nsf_hifigan.model.nsf_hifigan import NSF_HifiGAN

from nsf_hifigan.data.collate import MelCollate

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from nsf_hifigan.data.collate import MelCollate
from nsf_hifigan.model.generators.generator import Generator
from nsf_hifigan.hparams import HParams
from nsf_hifigan.data.dataset import MelDataset, MelDataset
import scipy.io.wavfile as wavfile
import librosa
from librosa import pyin
from nsf_hifigan.utils import load_filepaths, load_wav_to_torch
# from nsf_hifigan.model.pipeline import AudioPipeline,AudioPipeline0
from nsf_hifigan.mel_processing import wav2mel

def get_hparams(config_path: str) -> HParams:
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)
    return hparams

def last_checkpoint(path: str) -> Optional[str]:
    ckpt_path = None
    if os.path.exists(os.path.join(path, "lightning_logs")):
        versions = glob.glob(os.path.join(path, "lightning_logs", "version_*"))
        if len(list(versions)) > 0:
            last_ver = sorted(list(versions), key=lambda p: int(p.split("_")[-1]))[-1]
            last_ckpt = os.path.join(last_ver, "checkpoints/last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
    return ckpt_path

def get_train_params(args, hparams):
    devices = [int(n.strip()) for n in args.device.split(",")]
    ckpt_path = str(last_checkpoint(hparams.trainer.default_root_dir))
    dir_path = os.path.dirname(ckpt_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path if dir_path else 'cp', save_last=True, every_n_train_steps=2000, save_weights_only=False,
        monitor="valid/loss_spec_epoch", mode="min", save_top_k=5
    )
    earlystop_callback = EarlyStopping(monitor="valid/loss_spec_epoch", mode="min", patience=20)

    trainer_params = {
        "accelerator": args.accelerator,
        "callbacks": [checkpoint_callback, earlystop_callback],
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

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/24k.json", help='JSON file for configuration')
    parser.add_argument('-a', '--accelerator', type=str, default="gpu", help='training device')
    parser.add_argument('-d', '--device', type=str, default="0", help='training device ids')
    parser.add_argument('-n', '--num-nodes', type=int, default=1, help='training node number')
    args = parser.parse_args()

    hparams = get_hparams(args.config)
    # lightning_fabric.utilities.seed.seed_everything(hparams.train.seed)
    pl.utilities.seed.seed_everything(hparams.train.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    devices = [int(n.strip()) for n in args.device.split(",")]

    trainer_params = get_train_params(args, hparams)

    # one_dataset = MelDataset('1/1.txt', hparams.data)
    # collate_fn =  MelCollate()
    # one_loader = DataLoader(one_dataset,batch_size=1,collate_fn=collate_fn)
    # batch = next(iter(one_loader))
    # x_wav, x_wav_lengths = batch["x_wav_values"], batch["x_wav_lengths"]

    # audio_pipeline = AudioPipeline0(freq=24000,
    #                                 n_fft=512,
    #                                 n_mel=80,
    #                                 win_length=512,
    #                                 hop_length=128)
    # x_mel = audio_pipeline(x_wav.squeeze(1),aug=True)
    # x_mel_lengths = (x_wav_lengths / 128).long()

    # x_mel, x_mel_lengths = batch["x_mel_values"], batch["x_mel_lengths"]
    # x_pitch, x_pitch_lengths = batch["x_pitch_values"], batch["x_pitch_lengths"]
    # y_wav, y_wav_lengths = batch["y_wav_values"], batch["y_wav_lengths"]

    # x_mel, ids_slice = rand_slice_segments(x_mel, x_mel_lengths, 16384 // 128)
    # x_pitch = slice_segments(x_pitch.unsqueeze(1), ids_slice, 16384 // 128).squeeze(1) # slice
    # y_wav = slice_segments(y_wav, ids_slice * 128, 16384) # slice


    # x_mel, ids_slice = rand_slice_segments(x_mel, x_mel_lengths, 16384 // 128)
    # x_pitch = slice_segments(x_pitch.unsqueeze(1), ids_slice,  16384 // 128).squeeze(1) # slice
    # y_wav = slice_segments(y_wav, ids_slice *  128, 16384) # slice



    # y_wav_hat = model.net_g(x_mel, x_pitch)
    # y_wav_hat =y_wav_hat[:,:,:len(y_wav[0][0])]
    # y_spec = mel_spectrogram(
    #     y_wav.squeeze(1).float(),
    #     512,
    #     80,
    #     24000,
    #     128,
    #     512,
    #     center=False,
    #     complex=True
    # )
    # y_spec_hat = mel_spectrogram(
    #     y_wav_hat.squeeze(1).float(),
    #     512,
    #     80,
    #     24000,
    #     128,
    #     512,
    #     center=False,
    #     complex=True
    # )

    # model
    model = NSF_HifiGAN(**hparams).cuda()
    
    # # 读取预训练权重
    # pretrain_ckpt = torch.load('cp/model_ckpt_steps_1512000.ckpt',map_location="cpu")["state_dict"]
    # new_pretrain_ckpt = {}
    # # 修改 model_gen 中的键
    # for k, v in pretrain_ckpt['model_gen'].items():
    #     new_key = 'net_g.' + k
    #     new_pretrain_ckpt[new_key] = v
    # # 修改 model_disc 中的键
    # for k, v in pretrain_ckpt['model_disc'].items():
    #     if k.startswith('mpd.'):
    #         new_key = 'net_period_d.' + k[4:]
    #     elif k.startswith('msd.'):
    #         new_key = 'net_scale_d.' + k[4:]
    #     else:
    #         new_key = k
    #     new_pretrain_ckpt[new_key] = v
    # # 使用 model.load_state_dict 加载修改后的字典
    # model.load_state_dict(new_pretrain_ckpt,strict=True)

    # 读取myvocoder
    newtrain_ckpt = torch.load('logs_24k/lightning_logs/version_2/checkpoints/last.ckpt',map_location="cpu")["state_dict"]
    model.load_state_dict(newtrain_ckpt,strict=True)

    # mel = torch.from_numpy(np.load('mel_e2e.npy')).transpose(1,2).cuda()
    # f0 = torch.from_numpy(np.load('f0_e2e.npy')).cuda()

    npy = 'dataset/data_24k/050_042.npy'
    mel = torch.from_numpy(np.load(npy,allow_pickle=True)[1]).cuda() # (T,n_mel_bins) 
    mel = mel.squeeze(0).transpose(1,0) # (n_mel_bins,T) 
    f0 = torch.from_numpy(np.load(npy,allow_pickle=True)[2]).cuda()

    model.eval()
    with torch.no_grad():

        y_wav = model.net_g(mel,f0)
        
    print(y_wav)
    y_wav = y_wav.cpu().numpy()
    wavfile.write('050042_myvocoder_dif.wav', 24000, y_wav)

if __name__ == '__main__':
    main()