import torch
import torch.nn as nn
import torch.nn.functional as F


class FMSMaskNet(nn.Module):
    def __init__(self, n_fft, batch_norm=True):
        super().__init__()
        self.n_fft = n_fft
        self.fms = nn.Sequential(
            ResBlock(1, 8, batch_norm=batch_norm),
            FMS(8),
            ResBlock(8, 16, batch_norm=batch_norm),
            FMS(16),
            ResBlock(16, 8, batch_norm=batch_norm),
            FMS(8),
            nn.Conv2d(8, 1, 1)
        )

    def forward(self, x):
        window = torch.hann_window(self.n_fft).to(x.device)
        spectrum = torch.view_as_real(torch.stft(x, n_fft=self.n_fft, window=window, return_complex=True))
        magnitude = torch.sqrt((spectrum ** 2).sum(-1) + 1e-9).unsqueeze(1)
        mul_factor = F.softplus(self.fms(magnitude).squeeze(1))
        spectrum = mul_factor.unsqueeze(-1) * spectrum
        out = torch.istft(torch.view_as_complex(spectrum), n_fft=self.n_fft, window=window)
        return out


class FMS(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=num_channels, out_features=num_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        s = self.pooling(x).squeeze(2, 3)
        s = self.linear(s).unsqueeze(2).unsqueeze(3)
        s = self.activation(s)
        return s * x + s


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='same', negative_slope=0.3, batch_norm=True):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels) if batch_norm else nn.Identity(),
            nn.LeakyReLU(negative_slope=negative_slope)
            )
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels) if batch_norm else nn.Identity(),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        x = self.proj(x) + self.net(self.head(x))
        return x
