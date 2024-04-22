import json
import os
from pathlib import Path

import random
import librosa.util
from tqdm import tqdm

import torch
import torchaudio

import src.model as module_model
from src.datasets.utils import MelSpectrogram, MelSpectrogramConfig
from src.metric import WMOSMetric, CompositeEval


class Inferencer:
    def __init__(self, config, segment_size=None):
        self.logger = config.get_logger("test")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = config.init_obj(config["generator"], module_model)
        # self.logger.info(model)

        self.logger.info("Loading checkpoint: {} ...".format(config.resume))
        checkpoint = torch.load(config.resume, map_location=self.device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint.keys() else checkpoint
        if config["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        self.model = model.to(self.device)
        self.model.eval()

        self.mel_spec = MelSpectrogram(MelSpectrogramConfig())
        self.target_sr = config["preprocessing"]["sr"]

        self.wmos = WMOSMetric() if torch.cuda.is_available() else None
        self.composite_eval = CompositeEval(self.target_sr)

        self.segment_size = segment_size
        self.start_ind = 0

    def _load_audio(self, path: str):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.target_sr)

        return audio_tensor

    def _cut_audio(self, wav, new_ind=True):
        if self.segment_size is not None and wav.shape[-1] > self.segment_size:
            ind = random.randint(0, wav.shape[-1] - self.segment_size) if new_ind else self.start_ind
            wav = wav[:, ind:ind + self.segment_size]
            self.start_ind = ind

        return wav

    def denoise_audio(self, noisy_path: str, out_path: str = "result.wav"):
        noisy_audio = self._load_audio(noisy_path)
        noisy_audio = self._cut_audio(noisy_audio, new_ind=True)

        audio_len = noisy_audio.shape[-1]
        to_pad = self.closest_power_of_two(audio_len) - audio_len
        noisy_audio = torch.nn.functional.pad(noisy_audio, (0, to_pad))

        noisy_mel = self.mel_spec(noisy_audio).to(self.device)

        with torch.no_grad():
            gen_audio = self.model(noisy_mel, noisy_audio.unsqueeze(0).to(self.device))
        gen_audio = gen_audio.cpu().squeeze(1)
        gen_audio = gen_audio[:, :audio_len]
        if out_path is not None:
            torchaudio.save(out_path, gen_audio, self.target_sr)

        return gen_audio

    def denoise_dir(self, noisy_dir: str, out_dir: str = "output"):
        assert Path(noisy_dir).exists(), "invalid noisy_path"

        if not Path(out_dir).exists():
            Path(out_dir).mkdir(exist_ok=True, parents=True)

        files = sorted(os.listdir(noisy_dir))
        noisy_dir, out_dir = Path(noisy_dir), Path(out_dir)

        for file_name in tqdm(files, desc="Process file"):
            noisy_path = str(noisy_dir / file_name)
            out_path = str(out_dir / file_name)
            _ = self.denoise_audio(noisy_path, out_path)

    def validate_audio(self, noisy_path: str, clean_path: str, out_path: str = "result.wav", verbose=True):
        gen_audio = self.denoise_audio(noisy_path, out_path)
        clean_audio = self._load_audio(clean_path)
        clean_audio = self._cut_audio(clean_audio, new_ind=False)

        if self.segment_size is not None and clean_audio.shape[-1] > self.segment_size:
            ind = self.start_ind
            clean_audio = clean_audio[:, ind:ind + self.segment_size]

        result = {"file": noisy_path.split('/')[-1]}
        if self.wmos is not None:
            result["wv-mos"] = self.wmos(gen_audio.to(self.device))

        if noisy_path != clean_path:
            to_pad = clean_audio.shape[1] - gen_audio.shape[1]
            gen_audio = torch.nn.functional.pad(gen_audio, (0, to_pad))
            metrics = self.composite_eval(gen_audio, clean_audio)
            result.update(metrics)

        if verbose:
            for key, val in result.items():
                print(f"{key}: {val}")

        return result

    def validate_dir(self, noisy_dir: str, clean_dir: str, out_dir: str = "output", verbose=True):
        assert Path(noisy_dir).exists(), "invalid noisy dir"
        assert Path(clean_dir).exists(), "invalid clean dir"

        if not Path(out_dir).exists():
            Path(out_dir).mkdir(exist_ok=True, parents=True)

        files = sorted(os.listdir(noisy_dir))
        noisy_dir, clean_dir, out_dir = Path(noisy_dir), Path(clean_dir), Path(out_dir)

        results = []
        metrics_score = {}

        for file_name in tqdm(files, desc="Process file"):
            noisy_path = str(noisy_dir / file_name)
            clean_path = str(clean_dir / file_name)
            out_path = str(out_dir / file_name)
            result = self.validate_audio(noisy_path, clean_path, out_path, verbose=False)

            for key, val in result.items():
                if key != "file":
                    metrics_score[key] = metrics_score.get(key, 0.0) + val

            results.append(result)

        if verbose:
            for key, val in metrics_score.items():
                print(f"{key}: {val / len(files):.3f}")

        with (out_dir / "result.txt").open("w") as f:
            json.dump(results, f, indent=2)

    @staticmethod
    def closest_power_of_two(n):
        return 1 << (n - 1).bit_length()

