import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module, mean=0.0, std=0.01):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d) or isinstance(module, nn.Conv2d):
        module.weight.data.normal_(mean=mean, std=std)


def fix_shapes(x, skip, mode='crop'):
    assert mode in ['crop', 'pad']

    diff = skip.shape[-1] - x.shape[-1]
    if mode == 'crop':
        assert diff % 2 == 0
        diff = diff // 2
        return x, skip[..., diff:-diff]

    return F.pad(x, (0, diff)), skip


def stft(x, n_fft=800):
    window = torch.hann_window(n_fft)
    spectrum = torch.stft(x, n_fft=n_fft, window=window, return_complex=True)
    spectrum = torch.view_as_real(spectrum)

    magnitude = torch.sqrt(spectrum[..., 0] ** 2 + spectrum[..., 1] ** 2)
    phase = torch.atan2(spectrum[..., 1], spectrum[..., 0])

    return magnitude, phase


def istft(mag, phase, n_fft=800):
    mag = mag.unsqueeze(-1)
    phase = phase.unsqueeze(-1)

    spectrum = torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1).contiguous()
    spectrum = torch.view_as_complex(spectrum)
    window = torch.hann_window(n_fft)
    x = torch.istft(spectrum, n_fft=n_fft, window=window)

    return x
