import torch
import torch.nn as nn
import torch.nn.functional as F


class FMSMaskNet(nn.Module):
    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.fms = nn.Sequential(
            FMS(n_fft // 2 + 1),
            FMS(n_fft // 2 + 1),
            FMS(n_fft // 2 + 1)
        )

    def forward(self, x):
        window = torch.hann_window(self.n_fft).to(x.device)
        spectrum = torch.stft(x, n_fft=self.n_fft, window=window, return_complex=False)
        magnitude = torch.sqrt((spectrum ** 2).sum(-1) + 1e-9)
        # mul_factor = F.softplus(self.fms(magnitude))
        # spectrum = mul_factor.unsqueeze(-1) * spectrum
        spectrum = self.fms(magnitude)
        out = torch.istft(torch.view_as_complex(spectrum), n_fft=self.n_fft, window=window)
        return out


class FMS(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.linear = nn.Linear(in_features=num_channels, out_features=num_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        s = self.pooling(x).squeeze(2)
        s = self.linear(s).unsqueeze(2)
        s = self.activation(s)
        return s * x + s