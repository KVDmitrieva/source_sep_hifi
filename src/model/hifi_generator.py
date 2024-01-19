import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from src.model.utils import init_weights
from src.model.base_model import BaseModel


class HiFiGenerator(BaseModel):
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

    def forward(self, mel, **batch):
        x = self.prolog(mel)
        x = self.upsampler(x)
        x = self.epilog(x)
        return x


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
