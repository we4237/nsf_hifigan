import math
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import torchaudio

import logging

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

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

mel_basis = {}
hann_window = {}

def wav2mel(y, n_fft: int, num_mels: int, sampling_rate: int, hop_size: int, win_size: int,
                    center: bool = False, complex: bool = False):
    fmin = 30
    fmax = 12000

    y = y.clamp(min=-1., max=1.)
    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    if fmax not in mel_basis:
        mel = librosa.filters.mel(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[dtype_device] = torch.from_numpy(mel).float().to(y.device)

    x_stft = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                        pad_mode='constant', normalized=False, onesided=True, return_complex=True)
    spc = torch.abs(x_stft) 


    if complex:
        spec = torch.sqrt(spc.real.pow(2) + spc.imag.pow(2) + 1e-9)
        spec = torch.matmul(mel_basis[dtype_device], spec)
        mel = spectral_normalize_torch(spec)
    else:
        mel = torch.matmul(mel_basis[dtype_device], spc)
        eps = 1e-6
        mel = torch.log10(torch.clamp(mel, min=eps))
        
        # mel = mel.T  # (T,n_mel_bins) #输出的mel
        # mel = mel.transpose(1, 2)
    return mel # (n_bins, T)






def spectrogram_torch(y, n_fft: int, sampling_rate: int, hop_size: int, win_size: int, center: bool=False):
    if torch.min(y) < -1.:
        logging.warning(f'min value is {torch.min(y).detach().cpu().item()}')
    if torch.max(y) > 1.:
        logging.warning(f'max value is {torch.max(y).detach().cpu().item()}')

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)

    return spec


def spectrogram_torch_audio(y, n_fft: int,num_mels:int,  sampling_rate: int, hop_size: int, win_size: int, center: bool=False):
    if torch.min(y) < -1.:
        logging.warning(f'min value is {torch.min(y).detach().cpu().item()}')
    if torch.max(y) > 1.:
        logging.warning(f'max value is {torch.max(y).detach().cpu().item()}')
    fmin = 80
    fmax = 750
    global mel_basis,hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    pad = int((n_fft-hop_size)/2)
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[dtype_device] = torch.from_numpy(mel).float().to(y.device)
    spec = torchaudio.functional.spectrogram(y, pad, hann_window[wnsize_dtype_device],
            n_fft, hop_size, win_size, None,
            center=center, pad_mode='reflect', normalized=False, onesided=True)
    
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
    spec = torch.matmul(mel_basis[dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)

    if len(spec.shape) == 3:
        mel_matrix = mel_basis[fmax_dtype_device].unsqueeze(0)
    else:
        mel_matrix = mel_basis[fmax_dtype_device]
    spec = torch.matmul(mel_matrix, spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft: int, num_mels: int, sampling_rate: int, hop_size: int, win_size: int, fmin: int, fmax: int, center: bool=False):
    if torch.min(y) < -1.:
        logging.warning(f'min value is {torch.min(y).detach().cpu().item()}')
    if torch.max(y) > 1.:
        logging.warning(f'max value is {torch.max(y).detach().cpu().item()}')

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
