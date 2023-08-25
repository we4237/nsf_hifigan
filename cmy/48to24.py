import os
import soundfile as sf
import librosa

input_dir = "/mnt/user/chenmuyin/diffsinger/github/NSF-HiFiGAN-cmy/dataset_48k/data_1/wavs"
output_dir = "/mnt/user/chenmuyin/diffsinger/github/NSF-HiFiGAN-cmy/dataset/data_24k"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有的.wav文件路径
audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.wav')]

# 转换采样率并保存文件
for audio_file in audio_files:
    # 读取音频文件
    audio, sr = sf.read(audio_file)
    
    # 如果采样率已经是24000，则跳过该文件
    if sr == 24000:
        continue
    
    # 转换采样率
    audio_24k = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    
    # 获取输出文件名（添加_24k后缀）
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_file))[0] + '_24k.wav')
    
    # 保存文件
    sf.write(output_file, audio_24k, 24000, 'PCM_16')