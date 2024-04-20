import torch

from src.model.base_model import BaseModel
from src.model.hifi_generator import HiFiGenerator
from src.model.spectral import SpectralUNet, SpectralMaskNet
from src.model.wave import WaveUNet
from src.model.fms import FMSMaskNet
from src.model.utils import fix_shapes_1d, closest_power_of_two
from src.model.official_model import mel_spectrogram


class Generator(BaseModel):
    def __init__(self, generator_params, spectral_unet_params=None, wave_unet_params=None,
                 spectral_mask_params=None, add_spectral=False, concat_audio=True, use_fms=False):
        super().__init__()
        self.add_spectral = add_spectral
        self.concat_audio = concat_audio

        self.spec_unet = torch.nn.Identity() if spectral_unet_params is None else SpectralUNet(**spectral_unet_params)
        self.generator = HiFiGenerator(**generator_params)
        self.wave_unet = torch.nn.Identity() if wave_unet_params is None else WaveUNet(**wave_unet_params)

        mask_module = FMSMaskNet if use_fms else SpectralMaskNet
        self.spec_mask = torch.nn.Identity() if spectral_mask_params is None else mask_module(**spectral_mask_params)

    def forward(self, mel, audio=None, **batch):
        mel = self.get_melspec(audio.clone())

        mel_len = mel.shape[-1]
        pad_size = closest_power_of_two(mel.shape[-1]) - mel.shape[-1]
        mel = torch.nn.functional.pad(mel, (0, pad_size))

        spec_out = self.spec_unet(mel.unsqueeze(1))
        spec_out = spec_out[..., :mel_len]
        spec_out = spec_out.squeeze(1)
        gen_out = self.generator(spec_out)

        if audio is not None and self.concat_audio:
            gen_out = torch.cat(fix_shapes_1d(gen_out, audio), dim=1)

        wave_out = self.wave_unet(gen_out).squeeze(1)
        if self.add_spectral:
            return self.spec_mask(wave_out, spec_out).unsqueeze(1)

        return self.spec_mask(wave_out).unsqueeze(1)

    @staticmethod
    def get_melspec(x):
        shape = x.shape
        x = x.view(shape[0] * shape[1], shape[2])
        x = mel_spectrogram(x, 1024, 80, 16000, 256, 1024, 0, 8000)
        x = x.view(shape[0], -1, x.shape[-1])
        return x

