import torch
from torch import Tensor

from src.model.base_model import BaseModel
from src.model.hifi_generator import HiFiGenerator
from src.model.spectral import SpectralUNet, SpectralMaskNet
from src.model.wave import WaveUNet
from src.model.fms import FMSMaskNet
from src.model.utils import fix_shapes_1d, closest_power_of_two


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
        spec_out = self._apply_specrtal(mel)
        gen_out = self.generator(spec_out)

        if audio is not None and self.concat_audio:
            gen_out = torch.cat(fix_shapes_1d(gen_out, audio), dim=1)

        wave_out = self.wave_unet(gen_out).squeeze(1)
        if self.add_spectral:
            return self.spec_mask(wave_out, spec_out).unsqueeze(1)

        return self.spec_mask(wave_out).unsqueeze(1)
    
    def _apply_specrtal(self, mel):
        mel_len = mel.shape[-1]
        pad_size = closest_power_of_two(mel.shape[-1]) - mel.shape[-1]
        mel = torch.nn.functional.pad(mel, (0, pad_size))

        spec_out = self.spec_unet(mel.unsqueeze(1))
        spec_out = spec_out[..., :mel_len]
        spec_out = spec_out.squeeze(1)
        return spec_out
    

class ContextGenerator(Generator):
    def __init__(self, generator_params, spectral_unet_params=None, wave_unet_params=None,
                 spectral_mask_params=None, add_spectral=False, concat_audio=True, use_fms=False):
        super().__init__(generator_params, spectral_unet_params, wave_unet_params, 
                         spectral_mask_params, add_spectral, concat_audio, use_fms)


    def forward(self, mel: Tensor, audio: Tensor, context: Tensor = None, chunk_num: int = 1, **batch):
        # audio: batch * chunk_num x 1 x chunk_size
        spec_out = self._apply_specrtal(mel)
        gen_out = self.generator(spec_out)

        gen_out, audio = fix_shapes_1d(gen_out, audio)
        context_out = gen_out.detach()

        if context is None:
            bs, _, chunk_size = context_out.shape
            gen_context = context_out.reshape(-1, chunk_num, chunk_size)[:, :-1, :]
            cold_start = torch.zeros(bs // chunk_num, 1, chunk_size, device=audio.device)
            context = torch.cat([cold_start, gen_context], dim=1).reshape(bs, 1, chunk_size)

        assert gen_out.shape == audio.shape, "Shape mismatch (gen, audio)"
        assert gen_out.shape == context.shape, "Shape mismatch (gen, context)"

        gen_out = torch.cat([gen_out, audio, context], dim=1)

        wave_out = self.wave_unet(gen_out).squeeze(1)
        if self.add_spectral:
            return self.spec_mask(wave_out, spec_out).unsqueeze(1), context_out

        return self.spec_mask(wave_out).unsqueeze(1), context_out
