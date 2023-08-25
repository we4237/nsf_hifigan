
import logging
import librosa
from nsf_hifigan.mel_processing import wav2mel
import os
import torch
from tqdm import tqdm

MATPLOTLIB_FLAG = False
def plot_spectrogram_to_numpy(spectrogram,name):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.savefig(f'mel/{name}',dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    folder = 'infer_out/001_030'
    for wav in tqdm(os.listdir(folder)):
        wav_path = os.path.join(folder,wav)
        # wav_path = 'infer_out/001_000/001_000_24k.wav'
        name,_ = os.path.splitext(os.path.basename(wav_path))
        wav, _ =  librosa.core.load(wav_path, sr=24000)
        wav = torch.from_numpy(wav)
        spec = wav2mel(wav.squeeze(),
            512,
            80,
            24000,
            128,
            512,
            False,
            False)
        plot_spectrogram_to_numpy(spec,name)
