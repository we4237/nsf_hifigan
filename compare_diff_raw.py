import os
import numpy as np
import librosa
import parselmouth
from tqdm import tqdm

from compare import wav2melf0

path_list = np.loadtxt('filelists/24k_audio_filelist_train.txt', dtype=str)

for file_path in tqdm(path_list):
    mel_raw,f0_raw,mask = wav2melf0(file_path) # mel (n_mel_bins, T)
    mel_raw = mel_raw.transpose(1,0).unsqueeze(0).numpy()# mel (T,n_mel_bins)
    raw_mel_length = mel_raw.shape[1]
    f0_raw = f0_raw.numpy()
    # raw_wav_length = len(raw_wav)

    # npy_path = file_path.rsplit('_24k',1)[0] +'.npy'
    # raw_vocoder = np.load(npy_path,allow_pickle=True)[0]
    # mel_dif = np.load(npy_path,allow_pickle=True)[1]# mel (T,n_mel_bins)
    # dif_mel_length = mel_dif.shape[1]
    # f0_dif = np.load(npy_path,allow_pickle=True)[2]

    # print(f'raw wav lenght:{raw_wav_length}')
    # print(f'raw mel:{mel_raw.shape}')
    # print(f'raw mel max:{mel_raw.max()}')
    # print(f'raw mel min:{mel_raw.min()}')
    # print(f'raw f0:{f0_raw.shape}')
    # print(f'raw f0 max:{f0_raw.max()}')
    # print(f'raw f0 min:{f0_raw.min()}')

    # print(f'dif wav lenght:{diff_length}')
    # print(f'dif mel:{mel_dif.shape}')
    # print(f'dif mel max:{mel_dif.max()}')
    # print(f'dif mel min:{mel_dif.min()}')
    # print(f'dif f0:{f0_dif.shape}')
    # print(f'dif f0 max:{f0_dif.max()}')
    # print(f'dif f0 min:{f0_dif.min()}')

    # 如果raw长了就拿他补dif
    # if raw_mel_length > dif_mel_length:
    #     new_mel = np.zeros_like(mel_raw)
    #     new_mel[:,:dif_mel_length,:] = mel_dif
    #     new_mel[:,dif_mel_length:raw_mel_length,:] = mel_raw[:,dif_mel_length:,:]

    #     new_f0 = np.zeros_like(f0_raw)
    #     new_f0[:,:dif_mel_length] = f0_dif
    #     new_f0[:,dif_mel_length:raw_mel_length] = f0_raw[:,dif_mel_length:]
    # # elif raw_mel_length < dif_mel_length:
    # else:
    #     # 短了就切dif 
    #     new_mel = mel_dif[:,:raw_mel_length,:]
    #     new_f0 = f0_dif[:,:raw_mel_length]
    # # else:
    # #     continue
    # print(f'dif mel:{new_mel.shape}')


    # 保存raw mel和f0
    save_list = [mask,mel_raw,f0_raw]
    save_array = np.array(save_list)
    npy_path = file_path.rsplit('_24k',1)[0] + '_raw' +'.npy'
    np.save(npy_path,save_array)
