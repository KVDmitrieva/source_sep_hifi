import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

from src.model.base_model import BaseModel
from src.model.utils import init_weights


__all__ = ["Discriminator"]


class Discriminator(BaseModel):
    def __init__(self, pooling_params, sd_params, periods, pd_params):
        super().__init__()
        self.msd = nn.ModuleList([ScaleDiscriminator(p, **sd_params) for p in pooling_params])
        self.mpd = nn.ModuleList([PeriodDiscriminator(p, **pd_params) for p in periods])
        self.apply(init_weights)

    def forward(self, x, prefix="real", **batch):
        output = []
        feature_map = []
        for d in self.msd:
            out, f_map = d(x)
            output.append(out)
            feature_map.append(f_map)

        for d in self.mpd:
            out, f_map = d(x)
            output.append(out)
            feature_map.append(f_map)

        return {
            f"{prefix}_discriminator_out": output,
            f"{prefix}_feature_map": feature_map
        }


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
