import torch

from src.model.base_model import BaseModel
from src.model.hifi_generator import HiFiGenerator
from src.model.spectral import SpectralUNet, SpectralMaskNet
from src.model.wave import WaveUNet


class Generator(BaseModel):
    def __init__(self, spectral_unet_params, generator_params, wave_unet_params, spectral_mask_params, mode='vocoder'):
        super().__init__()
        self.spec_unet = SpectralUNet(**spectral_unet_params)
        self.generator = HiFiGenerator(**generator_params)
        self.wave_unet = WaveUNet(**wave_unet_params)
        self.spec_mask = SpectralMaskNet(**spectral_mask_params)
        self.mode = mode

    def forward(self, mel, audio=None):
        spec_out = self.spec_unet(mel.unsqueeze(1))
        gen_out = self.generator(spec_out)
        if audio is not None:
            gen_out = torch.cat([gen_out, audio], dim=1)

        wave_out = self.wave_unet(gen_out)
        mask_out = self.spec_mask(wave_out, spec_out) if self.mode == 'vocoder' else self.spec_mask(wave_out)
        return mask_out
