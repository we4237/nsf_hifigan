import torch
import numpy as np
import librosa
import parselmouth
from compare import wav2melf0

def estimate_pitch(audio: np.ndarray, mel:np.ndarray, sr: int, n_fft: int, win_length: int, hop_length: int,
                    method='parselmouth', normalize_mean=None, normalize_std=None, n_formants=1):
    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)


    if method == 'parselmouth':

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
        rpad = len(mel) - len(f0) - lpad if len(mel) - len(f0) - lpad != 0 else 0
        f0 = np.pad(f0, [[lpad, rpad]], mode='constant')

        f0 = f0[:len(mel)] #输出的f0
        pitch_mel = torch.from_numpy(f0).unsqueeze(0)
    else:
        raise ValueError

    pitch_mel = pitch_mel.float()


    return pitch_mel

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
    wav_path = "dataset/data_24k/001_030_24k.wav"
    wav, _ =  librosa.core.load(wav_path, sr=24000)


    npy = 'dataset/data_24k/001_030_raw.npy'

    audio_mel = np.load(npy,allow_pickle=True)[1] # (T,n_mel_bins) 
    mel = audio_mel.squeeze(0).transpose(1,0) # (n_mel_bins,T) 
    
    pitch_generate = np.load(npy,allow_pickle=True)[2]
    
    l_pad, r_pad = librosa_pad_lr(wav, 512, 128, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * 128]

    pitch_ = estimate_pitch(
        wav,
        mel,
        24000,
        512,
        512,
        128
    ).numpy()

    print(f'推理得到f0：{pitch_generate.sum()}')
    print(f'计算得到f0：{pitch_.sum()}')
