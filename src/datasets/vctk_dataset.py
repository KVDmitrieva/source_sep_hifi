import json
import logging
import os
import random
import shutil
from pathlib import Path

from typing import List

import torchaudio
from torch.nn.utils.rnn import pad_sequence
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from src.datasets.utils import MelSpectrogramConfig as config
from src.datasets.base_dataset import BaseDataset
from src.utils import ROOT_PATH


logger = logging.getLogger(__name__)

URL_LINKS = {
    "noisy_testset": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip",
    "clean_testset": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip",
    "noisy_trainset_28spk": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip",
    "clean_trainset_28spk": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip",
    "noisy_trainset_56spk": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_56spk_wav.zip",
    "clean_trainset_56spk": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_56spk_wav.zip",
}


class VCTKDataset(BaseDataset):
    def __init__(self, part, index_dir=None, data_dir=None, *args, **kwargs):
        assert part in ['testset', 'trainset_28spk', 'testset_56spk'] or part == 'train_all'

        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "datasets" / "vctk"
            index_dir.mkdir(exist_ok=True, parents=True)

        self._index_dir = Path(index_dir)
        self._data_dir = Path(index_dir) if data_dir is None else Path(data_dir)

        if part == 'train_all':
            index = sum([self._get_or_load_index(part) for part in ['trainset_28spk', 'testset_56spk']], [])
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

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

    @staticmethod
    def collate_fn(dataset_items: List[dict]):
        """
        Collate and pad fields in dataset items
        """
        spectrogram, audio = [], []
        target_spec, target_audio = [], []

        for item in dataset_items:
            audio.append(item["audio"].T)
            target_audio.append(item["target_audio"].T)
            spectrogram.append(item["spectrogram"].squeeze(0).T)
            target_spec.append(item["target_spectrogram"].squeeze(0).T)

        return {
            "audio": pad_sequence(audio, batch_first=True).transpose(1, 2),
            "target_audio": pad_sequence(target_audio, batch_first=True).transpose(1, 2),
            "mel": pad_sequence(spectrogram, batch_first=True, padding_value=config.pad_value).transpose(1, 2),
            "target_mel": pad_sequence(target_spec, batch_first=True, padding_value=config.pad_value).transpose(1, 2)
        }

    def _load_part(self, part):
        arch_path = self._index_dir / f"{part}.zip"
        print(f"Loading part {part}")

        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)

        os.remove(str(arch_path))

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"

        if not index_path.exists():
            self._create_index(part)

        with index_path.open() as f:
            index = json.load(f)

        return index

    def _create_index(self, part):
        index = []
        noisy_dir = self._data_dir / f"noisy_{part}_wav"
        clean_dir = self._data_dir / f"clean_{part}_wav"

        if not noisy_dir.exists():
            self._load_part(f"noisy_{part}")
            self._load_part(f"clean_{part}")

        wav_files = os.listdir(str(noisy_dir))

        for wav_file in tqdm(wav_files, desc=f"Preparing VCTK folders: {part}"):
            t_info = torchaudio.info(str(noisy_dir / wav_file))
            length = t_info.num_frames / t_info.sample_rate

            index.append(
                {
                    "noisy_path": str(noisy_dir / wav_file),
                    "clean_path": str(clean_dir / wav_file),
                    "audio_len": length
                }
            )

        with open(self._index_dir / f"{part}_index.json", "w") as f:
            json.dump(index, f, indent=2)
