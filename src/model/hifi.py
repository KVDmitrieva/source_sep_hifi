import torch.nn as nn
from torch.nn.utils import weight_norm

from src.model.base_model import BaseModel
from src.model.utils import *


__all__ = ["Generator", "Discriminator"]


class Generator(BaseModel):
    def __init__(self, prolog_params, upsampler_blocks_params, epilog_params):
        super().__init__()
        self.prolog = weight_norm(nn.Conv1d(**prolog_params))

        upsamplers = []
        for upsampler_params in upsampler_blocks_params:
            upsamplers.append(UpsamplerBlock(**upsampler_params))

        self.upsampler = nn.Sequential(*upsamplers)
        self.epilog = nn.Sequential(
            nn.LeakyReLU(),
            weight_norm(nn.Conv1d(**epilog_params)),
            nn.Tanh()
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.prolog(x)
        x = self.upsampler(x)
        x = self.epilog(x)
        return {
            "generator_audio": x
        }


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
