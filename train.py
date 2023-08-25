import os
import json
import glob
import argparse
from typing import Optional
import torch
import torchaudio
import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from nsf_hifigan.model.nsf_hifigan import NSF_HifiGAN

from nsf_hifigan.data.collate import MelCollate

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler,AdvancedProfiler

from nsf_hifigan.hparams import HParams
from nsf_hifigan.data.dataset_06 import MelDataset

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
    checkpoint_callback = ModelCheckpoint(
        dirpath=None, save_last=True, every_n_train_steps=2000, save_weights_only=False,
        monitor="valid/loss_spec_epoch", mode="min", save_top_k=20
    )
    # earlystop_callback = EarlyStopping(monitor="valid/loss_spec_epoch", mode="min", patience=1000)

    trainer_params = {
        "accelerator": args.accelerator,
        "callbacks": [checkpoint_callback],
    }

    if args.accelerator != "cpu":
        trainer_params["devices"] = devices

    if len(devices) > 1:
        trainer_params["strategy"] = "ddp"

    trainer_params.update(hparams.trainer)

    if hparams.train.fp16_run:
        trainer_params["amp_backend"] = "native"
        trainer_params["precision"] = 16
    else:
        trainer_params["amp_backend"] = "native"
        trainer_params["precision"] = 32
    
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    devices = [int(n.strip()) for n in args.device.split(",")]

    # data
    train_dataset = MelDataset(hparams.data.training_files, hparams.data)
    valid_dataset = MelDataset(hparams.data.validation_files, hparams.data)

    collate_fn = MelCollate()

    trainer_params = get_train_params(args, hparams)

    if "strategy" in trainer_params and trainer_params["strategy"] == "ddp":
        batch_per_gpu = hparams.train.batch_size // len(devices)
    else:
        batch_per_gpu = hparams.train.batch_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_per_gpu, num_workers=8, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=4, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    # model
    model = NSF_HifiGAN(**hparams)
    
    # 读取预训练权重
    # state = ckpt_dict["state_dict"]["model_gen"]
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
    # 使用 model.load_state_dict 加载修改后的字典
    model.load_state_dict(new_pretrain_ckpt,strict=True)
    trainer = pl.Trainer(**trainer_params)
    
    # pretrain_ckpt = torch.load('logs_bak/float32/version_7/checkpoints/last.ckpt',map_location="cpu")["state_dict"]
    # model.load_state_dict(pretrain_ckpt,strict=True)

    # resume training
    ckpt_path = last_checkpoint(hparams.trainer.default_root_dir)
    # ckpt_path = 'logs_bak/version_0628_difmel+diff0/checkpoints/last.ckpt'
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
  main()
