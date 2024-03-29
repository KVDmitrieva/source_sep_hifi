import logging
import random
from typing import List

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.parse_config import ConfigParser
from src.datasets.utils import MelSpectrogram, MelSpectrogramConfig

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            spec_augs=None,
            limit=None,
            max_audio_length=None
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        self.log_spec = config_parser["preprocessing"]["log_spec"]
        self.mel_spec = MelSpectrogram(MelSpectrogramConfig())

        self.max_len = max_audio_length
        index = self._filter_records_from_dataset(index, limit)
        self._index: List[dict] = index

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        target_sr = self.config_parser["preprocessing"]["sr"]
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

    @staticmethod
    def collate_fn():
        raise NotImplementedError()
