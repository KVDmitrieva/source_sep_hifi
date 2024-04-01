import torch
from torch import nn

import torchaudio
import librosa


class MelSpectrogram(nn.Module):

    def __init__(self, sr, win_length, hop_length, n_fft, f_min, f_max, n_mels, power, pad_value):
        super(MelSpectrogram, self).__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            pad=(win_length - hop_length) // 2,
            center=False
        )

        self.mel_spectrogram.spectrogram.power = power

        mel_basis = librosa.filters.mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))
        self.pad_value = pad_value
        self.sr = sr

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio).clamp_(min=1e-5).log_()
        return mel
