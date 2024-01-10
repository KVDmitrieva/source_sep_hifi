import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F


__all__ = ["UpsamplerBlock", "ScaleDiscriminator", "PeriodDiscriminator", "init_weights"]


class UpsamplerBlock(nn.Module):
    def __init__(self, upsampler_params, res_block_kernels=(3, 7, 11), res_block_dilation=((1, 1), (3, 1), (5, 1))):
        super().__init__()
        self.upsampler = nn.Sequential(
            nn.LeakyReLU(),
            weight_norm(nn.ConvTranspose1d(**upsampler_params))
        )
        self.n = len(res_block_kernels)
        res_blocks = []
        for i in range(self.n):
            res_blocks.append(ResStack(upsampler_params["out_channels"], res_block_kernels[i], res_block_dilation))

        self.mfr = nn.ModuleList(res_blocks)

    def forward(self, x):
        x = self.upsampler(x)
        mfr_out = torch.zeros_like(x, device=x.device)
        for res_block in self.mfr:
            mfr_out = mfr_out + res_block(x)
        return mfr_out / self.n


class ResStack(nn.Module):
    def __init__(self, channels_num, kernel_size, dilation: list):
        super().__init__()
        n = len(dilation)
        net = []
        for i in range(n):
            net.append(nn.Sequential(
                nn.LeakyReLU(),
                weight_norm(nn.Conv1d(in_channels=channels_num, out_channels=channels_num,
                                      kernel_size=kernel_size, dilation=dilation[i][0], padding='same')),
                nn.LeakyReLU(),
                weight_norm(nn.Conv1d(in_channels=channels_num, out_channels=channels_num,
                                      kernel_size=kernel_size, dilation=dilation[i][1], padding='same'))
            ))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for block in self.net:
            x = x + block(x)
        return x


class ScaleDiscriminator(nn.Module):
    def __init__(self, pooling, prolog_params, downsampler_params, post_downsampler_params, epilog_params):
        super().__init__()
        norm = spectral_norm if pooling == 1 else weight_norm
        self.pooling = nn.AvgPool1d(pooling, padding=pooling // 2)
        self.prolog = norm(nn.Conv1d(**prolog_params))
        self.downsampler = nn.ModuleList([norm(nn.Conv1d(**params)) for params in downsampler_params])
        self.post_downsampler = norm(nn.Conv1d(**post_downsampler_params))
        self.epilog =  norm(nn.Conv1d(**epilog_params))

    def forward(self, x):
        x = self.pooling(x)

        feature_maps = []
        x = F.leaky_relu(self.prolog(x))
        feature_maps.append(x)

        for downsampler in self.downsampler:
            x = F.leaky_relu(downsampler(x))
            feature_maps.append(x)

        x = F.leaky_relu(self.post_downsampler(x))
        feature_maps.append(x)

        x = self.epilog(x)
        feature_maps.append(x)
        return torch.flatten(x, 1), feature_maps


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, stem_params, poststem_params, epilog_params):
        super().__init__()
        self.period = period
        self.stem = nn.ModuleList([weight_norm(nn.Conv2d(**stem_param)) for stem_param in stem_params])
        self.post_stem = weight_norm(nn.Conv2d(**poststem_params))
        self.epilog = weight_norm(nn.Conv2d(**epilog_params))

    def forward(self, x):
        batch_size, channels, len_t = x.shape
        mod = len_t % self.period
        pad_len = 0 if mod == 0 else self.period - mod
        x = F.pad(x, (0, pad_len))
        batch_size, channels, len_t = x.shape
        x = x.reshape(batch_size, channels, len_t // self.period, self.period)

        feature_maps = []
        for stem in self.stem:
            x = F.leaky_relu(stem(x))
            feature_maps.append(x)

        x = F.leaky_relu(self.post_stem(x))
        feature_maps.append(x)

        x = self.epilog(x)
        feature_maps.append(x)

        return torch.flatten(x), feature_maps


def init_weights(module, mean=0.0, std=0.01):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d) or isinstance(module, nn.Conv2d):
        module.weight.data.normal_(mean=mean, std=std)
