import logging
import random
from typing import List

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hydra.utils import instantiate

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(self, index, config, wave_augs=None, spec_augs=None, limit=None, max_audio_length=None):
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        self.log_spec = config.preprocessing.log_spec
        self.mel_spec = instantiate(config.preprocessing.spectrogram)

        self.max_len = max_audio_length
        index = self._filter_records_from_dataset(index, limit)
        self._index: List[dict] = index

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        noisy_audio = self.load_audio(data_dict["noisy_path"])
        clean_audio = self.load_audio(data_dict["clean_path"])

        if self.max_len is not None and noisy_audio.shape[-1] > self.max_len:
            ind = random.randint(0, noisy_audio.shape[-1] - self.max_len)
            noisy_audio = noisy_audio[:, ind:ind + self.max_len]
            clean_audio = clean_audio[:, ind:ind + self.max_len]
        noisy_audio, noisy_spec = self.process_wave(noisy_audio)
        clean_audio, clean_spec = self.process_wave(clean_audio)
        return {
            "audio": noisy_audio,
            "spectrogram": noisy_spec,
            "target_audio": clean_audio,
            "target_spectrogram": clean_spec
        }

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        target_sr = self.mel_spec.sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)

            audio_tensor_spec = self.mel_spec(audio_tensor_wave)
            if self.spec_augs is not None:
                audio_tensor_spec = self.spec_augs(audio_tensor_spec)
            return audio_tensor_wave, audio_tensor_spec

    @staticmethod
    def _filter_records_from_dataset(index: list, limit) -> list:
        if limit is not None:
            random.seed(42)
            random.shuffle(index)
            index = index[:limit]
        return index
