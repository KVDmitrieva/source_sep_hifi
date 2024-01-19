import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.utils import fix_shapes_2d, stft, istft


class SpectralMaskNet(nn.Module):
    def __init__(self, n_fft, spectral_params):
        super().__init__()
        self.n_fft = n_fft
        self.spectral = SpectralUNet(**spectral_params)

        self.spec_conv = nn.Conv1d(80, 513, 1)
        self.reflection = nn.ReflectionPad1d((1, 0))


    def forward(self, x, spectral_out=None):
        print("stft", x.shape, spectral_out.shape)
        magnitude, phase = stft(x, self.n_fft)
        mag = magnitude.unsqueeze(1)
        print("mag", mag.shape)
        if spectral_out is not None:
            spectral_out = self.reflection(spectral_out)
            spectral_out = self.spec_conv(spectral_out)
            mag = torch.cat([mag, spectral_out.unsqueeze(1)], dim=1)
        print("magnitude", x.shape, spectral_out.shape, mag.shape)
        # mag = magnitude if spectral_out is None else

        mul_factor = F.softplus(self.spectral(mag))
        magnitude = magnitude * mul_factor

        out = istft(magnitude, phase, self.n_fft)
        return out


class SpectralUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: list,
                 scale: int, bottleneck_width: int, block_depth=4, kernel_size=3, padding=1):
        super().__init__()
        self.prolog = nn.Conv2d(in_channels=in_channels, out_channels=width[0], kernel_size=kernel_size, padding=padding)
        self.downsample = nn.ModuleList([DownsampleBlock(w, scale, block_depth, kernel_size, padding) for w in width])
        self.upsample = nn.ModuleList([UpsampleBlock(w, scale, block_depth, kernel_size, padding) for w in width[::-1]])
        self.bottleneck = Bottleneck(bottleneck_width, block_depth, kernel_size, padding)
        self.epilog = nn.Conv2d(in_channels=width[0], out_channels=out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        skips = []
        out = self.prolog(x)
        for block in self.downsample:
            print(out.shape)
            out, skip = block(out)
            skips.append(skip)

        out = self.bottleneck(out)
        for block, skip in zip(self.upsample, skips[::-1]):
            out = block(out, skip)

        return self.epilog(out)


class UpsampleBlock(nn.Module):
    def __init__(self, width, scale, block_depth, block_kernel, block_padding):
        super().__init__()
        self.prolog = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='nearest'),
            nn.Conv2d(in_channels=2 * width, out_channels=width, kernel_size=1)
        )
        self.blocks = nn.Sequential(*[BlockWidth(width, block_kernel, block_padding) for _ in range(block_depth)])
        self.epilog = nn.Conv2d(in_channels=2 * width, out_channels=width, kernel_size=1)

    def forward(self, x, skip):
        x = self.prolog(x)
        x = self.blocks(x)
        # x, skip = fix_shapes_2d(x, skip, mode='pad')
        x = self.epilog(torch.cat([x, skip], dim=1))
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, width, scale, block_depth, block_kernel, block_padding):
        super().__init__()
        self.blocks = nn.Sequential(*[BlockWidth(width, block_kernel, block_padding) for _ in range(block_depth)])
        self.epilog = nn.Conv2d(in_channels=width, out_channels=2 * width, kernel_size=scale, stride=scale)

    def forward(self, x):
        out = self.blocks(x)
        out = self.epilog(out)
        return out, x


class Bottleneck(nn.Module):
    def __init__(self, width, block_depth, block_kernel, block_padding):
        super().__init__()
        self.blocks = nn.Sequential(*[BlockWidth(width, block_kernel, block_padding) for _ in range(block_depth)])

    def forward(self, x):
        return self.blocks(x)


class BlockWidth(nn.Module):
    def __init__(self, width, kernel_size, padding):
        super().__init__()
        self.block = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return x + self.block(F.leaky_relu(x))
