import torch

from src.model.base_model import BaseModel
from src.model.hifi_generator import HiFiGenerator
from src.model.spectral import SpectralUNet, SpectralMaskNet
from src.model.wave import WaveUNet
from src.model.utils import fix_shapes_1d


class Generator(BaseModel):
    def __init__(self, generator_params, spectral_unet_params=None, wave_unet_params=None,
                 spectral_mask_params=None, add_spectral=False, concat_audio=True):
        super().__init__()
        self.add_spectral = add_spectral
        self.concat_audio = concat_audio

        self.spec_unet = torch.nn.Identity() if spectral_unet_params is None else SpectralUNet(**spectral_unet_params)
        self.generator = HiFiGenerator(**generator_params)
        self.wave_unet = torch.nn.Identity() if wave_unet_params is None else WaveUNet(**wave_unet_params)
        self.spec_mask = torch.nn.Identity() if spectral_mask_params is None else SpectralMaskNet(**spectral_mask_params)

    def forward(self, mel, audio=None, **batch):
        spec_out = self.spec_unet(mel.unsqueeze(1))
        spec_out = spec_out.squeeze(1)
        gen_out = self.generator(spec_out)

        if audio is not None and self.concat_audio:
            gen_out = torch.cat([gen_out, fix_shapes_1d(gen_out, audio)], dim=1)

        wave_out = self.wave_unet(gen_out).squeeze(1)
        if self.add_spectral:
            return self.spec_mask(wave_out, spec_out)

        return self.spec_mask(wave_out)

