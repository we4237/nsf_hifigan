import librosa
import numpy as np #1.20
import parselmouth

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
    
if __name__ =='__main__':
    fft_size=512
    hop_size=128
    win_length=512
    window="hann"
    num_mels=80
    fmin=30
    fmax=12000
    eps=1e-6
    sample_rate=24000
    loud_norm=False
    min_level_db=-120
    return_linear=False
    trim_long_sil=False

    wav_path = "dataset/data_24k/001_000_24k.wav"
    wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                            win_length=win_length, window=window, pad_mode="constant")
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc

    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)

    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]
    mel = mel.T  # (T,n_mel_bins) #输出的mel
    # print(mel.sum())

    time_step = hop_size / sample_rate * 1000
    f0_min = 80
    f0_max = 750
    if hop_size == 128:
        pad_size = 4
    elif hop_size == 256:
        pad_size = 2
    else:
        assert False

    f0 = parselmouth.Sound(wav, sample_rate).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    lpad = pad_size * 2
    rpad = len(mel) - len(f0) - lpad
    f0 = np.pad(f0, [[lpad, rpad]], mode='constant')

    f0 = f0[:len(mel)] #输出的f0
    print(f0.sum())