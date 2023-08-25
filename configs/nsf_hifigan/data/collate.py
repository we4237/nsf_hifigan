
import torch

class MelCollate():
    def __init__(self, return_ids: bool = False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x["wav_raw"].size(1) for x in batch]),
            dim=0, descending=True)

        max_x_wav_len = max([x["wav_raw"].size(1) for x in batch])
        max_x_mel_len1 = max([x["mel"].size(0) for x in batch])
        max_x_mel_len2 = max([x["mel"].size(1) for x in batch])
        max_x_pitch_len = max([x["pitch"].size(1) for x in batch])
        max_y_wav_len = max([x["wav_padded"].size(1) for x in batch])

        x_wav_lengths = torch.LongTensor(len(batch))
        x_mel_lengths = torch.LongTensor(len(batch))
        x_pitch_lengths = torch.LongTensor(len(batch))
        y_wav_lengths = torch.LongTensor(len(batch))

        x_wav_padded = torch.zeros(len(batch), 1, max_x_wav_len, dtype=torch.float32)
        x_mel_padded = torch.zeros(len(batch), max_x_mel_len1 , max_x_mel_len2, dtype=torch.float32)
        x_pitch_padded = torch.zeros(len(batch), max_x_pitch_len, dtype=torch.float32)
        y_wav_padded = torch.zeros(len(batch), 1, max_y_wav_len, dtype=torch.float32)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            wav = row["wav_raw"]
            x_wav_padded[i, :, :wav.size(1)] = wav
            x_wav_lengths[i] = wav.size(1)
            
            pitch = row["pitch"]
            x_pitch_padded[i, :pitch.size(1)] = pitch
            x_pitch_lengths[i] = pitch.size(1)-1

            mel = row["mel"]
            x_mel_padded[i,:mel.size(0), :mel.size(1)] = mel
            x_mel_lengths[i] = mel.size(1)-1

            wav = row["wav_padded"]
            y_wav_padded[i, :, :wav.size(1)] = wav
            y_wav_lengths[i] = wav.size(1)

        ret = {
            "x_wav_values": x_wav_padded,
            "x_wav_lengths": x_wav_lengths,
            "x_mel_values": x_mel_padded,
            "x_mel_lengths": x_mel_lengths,
            "x_pitch_values": x_pitch_padded,
            "x_pitch_lengths": x_pitch_lengths,
            "y_wav_values": y_wav_padded,
            "y_wav_lengths": y_wav_lengths,
        }

        if self.return_ids:
            ret.update("ids", "ids_sorted_decreasing")
        return ret

