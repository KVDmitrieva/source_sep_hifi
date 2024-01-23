import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module, mean=0.0, std=0.01):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d) or isinstance(module, nn.Conv2d):
        module.weight.data.normal_(mean=mean, std=std)


def fix_shapes_1d(x, skip, mode='crop'):
    assert mode in ['crop', 'pad']

    diff = skip.shape[-1] - x.shape[-1]
    if mode == 'crop':
        assert diff % 2 == 0
        diff = diff // 2
        return x, skip[..., diff:-diff]

    return F.pad(x, (0, diff)), skip


def fix_shapes_2d(x, skip, mode='crop'):
    assert mode in ['crop', 'pad']

    diff2 = skip.shape[-2] - x.shape[-2]
    diff1 = skip.shape[-1] - x.shape[-1]
    if mode == 'crop':
        diff1 = diff1 // 2
        diff2 = diff2 // 2
        return x, skip[..., diff2:-diff2, diff1:-diff1]

    return F.pad(x, (0, diff1, 0, diff2)), skip


def stft(x, n_fft=800):
    window = torch.hann_window(n_fft).to(x.device)
    spectrum = torch.stft(x, n_fft=n_fft, window=window, return_complex=True)
    spectrum = torch.view_as_real(spectrum)

    magnitude = torch.sqrt(spectrum[..., 0] ** 2 + spectrum[..., 1] ** 2)
    phase = torch.atan2(spectrum[..., 1], spectrum[..., 0])

    return magnitude, phase


def istft(mag, phase, n_fft=800):
    spectrum = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    window = torch.hann_window(n_fft).to(spectrum.device)
    x = torch.istft(spectrum, n_fft=n_fft, window=window)

    return x
