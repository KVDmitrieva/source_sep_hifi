import json
import logging
import os
import shutil
import random
from pathlib import Path

from tqdm import tqdm
from typing import List

import torchaudio
from torch.nn.utils.rnn import pad_sequence
from speechbrain.utils.data_utils import download_file

from src.datasets.utils import MelSpectrogramConfig as config
from src.datasets.base_dataset import BaseDataset
from src.utils import ROOT_PATH


logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
}


class LJspeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["noisy_path"]
        audio_wave = self.load_audio(audio_path)
        if audio_wave.shape[-1] > self.max_len:
            ind = random.randint(0, audio_wave.shape[-1] - self.max_len)
            audio_wave = audio_wave[:, ind:ind + self.max_len]
        audio_wave, audio_spec = self.process_wave(audio_wave)
        return {
            "audio": audio_wave,
            "spectrogram": audio_spec
        }

    @staticmethod
    def collate_fn(dataset_items: List[dict]):
        """
        Collate and pad fields in dataset items
        """
        spectrogram, audio = [], []

        for item in dataset_items:
            audio.append(item["audio"].T)
            spectrogram.append(item["spectrogram"].squeeze(0).T)

        return {
            "audio": pad_sequence(audio, batch_first=True).transpose(1, 2),
            "mel": pad_sequence(spectrogram, batch_first=True, padding_value=config.pad_value).transpose(1, 2)
        }

    def _load_dataset(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

        files = [file_name for file_name in (self._data_dir / "wavs").iterdir()]
        train_length = int(0.85 * len(files)) # hand split, test ~ 15%
        (self._data_dir / "train").mkdir(exist_ok=True, parents=True)
        (self._data_dir / "test").mkdir(exist_ok=True, parents=True)
        for i, fpath in enumerate((self._data_dir / "wavs").iterdir()):
            if i < train_length:
                shutil.move(str(fpath), str(self._data_dir / "train" / fpath.name))
            else:
                shutil.move(str(fpath), str(self._data_dir / "test" / fpath.name))
        shutil.rmtree(str(self._data_dir / "wavs"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_dataset()

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(
                list(wav_dirs), desc=f"Preparing ljspeech folders: {part}"
        ):
            wav_dir = Path(wav_dir)
            trans_path = list(self._data_dir.glob("*.csv"))[0]
            with trans_path.open() as f:
                for line in f:
                    w_id = line.split('|')[0]
                    w_text = " ".join(line.split('|')[1:]).strip()
                    wav_path = wav_dir / f"{w_id}.wav"
                    if not wav_path.exists(): # elem in another part
                        continue
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    if w_text.isascii():
                        index.append(
                            {
                                "noisy_path": str(wav_path.absolute().resolve()),
                                "text": w_text.lower(),
                                "audio_len": length,
                            }
                        )
        return index
