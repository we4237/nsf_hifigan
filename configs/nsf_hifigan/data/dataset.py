import os
import random
from typing import Optional

import torch
import torchaudio

import numpy as np
import librosa
from librosa import pyin
import parselmouth
from ..utils import load_filepaths, load_wav_to_torch
from ..model.pipeline import AudioPipeline

resamplers = {}

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

def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch

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

def estimate_pitch(audio: np.ndarray, mel:np.ndarray, sr: int, n_fft: int, win_length: int, hop_length: int,
                    method='parselmouth', normalize_mean=None, normalize_std=None, n_formants=1):
    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'pyin':
        snd, sr = audio, sr
        pad_size = int((n_fft-hop_length)/2)
        snd = np.pad(snd, (pad_size, pad_size), mode='reflect')

        pitch_mel, voiced_flag, voiced_probs = pyin(
            snd,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=win_length,
            hop_length=hop_length,
            center=False,
            pad_mode='reflect')
        # assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        # pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError
        
    elif method == 'parselmouth':

        mel = mel.T # (T,n_mel_bins) 

        time_step = hop_length / sr * 1000
        f0_min = 80
        f0_max = 750
        if hop_length == 128:
            pad_size = 4
        elif hop_length == 256:
            pad_size = 2
        else:
            assert False
                
        f0 = parselmouth.Sound(audio, sr).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
        lpad = pad_size * 2
        rpad = len(mel) - len(f0) - lpad
        f0 = np.pad(f0, [[lpad, rpad]], mode='constant')

        f0 = f0[:len(mel)] # 输出的f0
        pitch_mel = torch.from_numpy(f0).unsqueeze(0)
    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel

def get_pitch(audio: str,mel: str, sr: int, filter_length: int, win_length: int, hop_length: int):
    pitch_mel = estimate_pitch(
        audio=audio, mel=mel, sr=sr, n_fft=filter_length,
        win_length=win_length, hop_length=hop_length, method='parselmouth',
        normalize_mean=None, normalize_std=None, n_formants=1)

    return pitch_mel

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, audiopaths: str, hparams):
        self.audiopaths = load_filepaths(audiopaths)
        self.hparams = hparams
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.mel_fmin       = hparams.mel_fmin
        self.mel_fmax       = hparams.mel_fmax
        self.n_mel_channels = hparams.n_mel_channels

        self.resamplers = {}

        random.seed(1234)
        random.shuffle(self.audiopaths)

        self.audio_pipeline = AudioPipeline(sample_rate=self.sampling_rate,
                                            fft_size=self.filter_length,
                                            num_mels=self.n_mel_channels,
                                            win_length=self.win_length,
                                            hop_size=self.hop_length)
        for param in self.audio_pipeline.parameters():
            param.requires_grad = False

    def get_item(self, index: int):
        audio_path = self.audiopaths[index]
        
        audio_wav = load_audio(audio_path, sr=self.sampling_rate)
        # 原始数据mel
        with torch.inference_mode():
            wav,audio_mel = self.audio_pipeline(audio_wav)    

        # 生成数据f0
        npy = audio_path.rsplit('_24k', 1)[0] + '.npy' 
        # audio_mel = torch.from_numpy(np.load(npy,allow_pickle=True)[1]) # (T,n_mel_bins) 
        # audio_mel = audio_mel.squeeze(0).transpose(1,0) # (n_mel_bins,T) 
        audio_pitch = torch.from_numpy(np.load(npy,allow_pickle=True)[2])

        audio_pitch = get_pitch(
            wav.numpy(),
            audio_mel.numpy(),
            self.sampling_rate,
            self.filter_length,
            self.win_length,
            self.hop_length
        )

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
