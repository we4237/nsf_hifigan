import os
import random
import re
from typing import Optional

import torch
import torchaudio

import numpy as np

from compare import wav2melf0

from ..utils import load_filepaths, load_wav_to_torch


resamplers = {}
def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2
    
def load_audio(filename: str, sr: Optional[int] = None):
    global resamplers
    audio, sampling_rate = load_wav_to_torch(filename)

    if sr is not None and sampling_rate != sr:
        # not match, then resample
        if sr in resamplers:
            resampler = resamplers[(sampling_rate, sr)]
        else:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=sr)
            resamplers[(sampling_rate, sr)] = resampler
        audio = resampler(audio)
        sampling_rate = sr
        # raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))
    return audio

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, audiopaths: str, hparams):
        self.audiopaths = load_filepaths(audiopaths)
        self.hparams = hparams
        self.sampling_rate  = hparams.sampling_rate
        self.fft_size  = hparams.filter_length
        self.hop_size     = hparams.hop_length

        self.resamplers = {}

        random.seed(1234)
        random.shuffle(self.audiopaths)

    def get_item(self, index: int):
        audio_path = self.audiopaths[index]
        
        audio_wav = load_audio(audio_path, sr=self.sampling_rate)
        
        # npy = audio_path.rsplit('_24k',1)[0] + '_raw' +'.npy'
        npy = audio_path.rsplit('_24k', 1)[0] + '.npy' 
        audio_mel = torch.from_numpy(np.load(npy,allow_pickle=True)[1]) # (T,n_mel_bins) 
        audio_mel = audio_mel.squeeze(0).transpose(1,0) # (n_mel_bins,T) 
        audio_pitch = torch.from_numpy(np.load(npy,allow_pickle=True)[2])

        # pt = audio_path.rsplit('_24k',1)[0] + '_dif' +'.pt'

        return {
            "wav_raw": audio_wav.unsqueeze(0),
            "mel": audio_mel,
            "pitch": audio_pitch,
            "wav_padded":audio_wav.unsqueeze(0),
        }

    def __getitem__(self, index):
        ret = self.get_item(index)
        return ret

    def __len__(self):
        return len(self.audiopaths)
