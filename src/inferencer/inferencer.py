import json
import os
from pathlib import Path

from tqdm import tqdm

import torch
import torchaudio

import src.model as module_model
from src.datasets.utils import MelSpectrogram, MelSpectrogramConfig
from src.metric import *


class Inferencer:
    def __init__(self, config):
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
        self.metrics = self._prepare_metrics()

    @staticmethod
    def _prepare_metrics():
        metric = {}
        if torch.cuda.is_available():
            metric["WMOS"] = WMOSMetric()

        metric["PESQ"] = PESQMetric()
        metric["SI-SDR"] = SISDRMetric()
        metric["SDR"] = SDRMetric()
        metric["STOI"] = STOIMetric()

        return metric

    def _load_audio(self, path: str):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.target_sr)

        return audio_tensor

    def denoise_audio(self, noisy_path: str, out_path: str = "result.wav"):
        noisy_audio = self._load_audio(noisy_path)
        noisy_mel = self.mel_spec(noisy_audio).to(self.device)

        with torch.no_grad():
            gen_audio = self.model(noisy_mel, noisy_audio.unsqueeze(0).to(self.device))
        gen_audio = gen_audio.cpu().squeeze(1)
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

        result = {"file": noisy_path.split('/')[-1]}
        for m in self.metrics.keys():
            if m == "WMOS":
                result[m] = self.metrics[m](gen_audio.to(self.device))
            else:
                to_pad = clean_audio.shape[1] - gen_audio.shape[1]
                gen_audio = torch.nn.functional.pad(gen_audio, (0, to_pad))
                result[m] = self.metrics[m](gen_audio, clean_audio).item()

            if verbose:
                print(f"{m}: {result[m]:.3f}")

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
        for m in self.metrics.keys():
            metrics_score[m] = 0.

        for file_name in tqdm(files, desc="Process file"):
            noisy_path = str(noisy_dir / file_name)
            clean_path = str(clean_dir / file_name)
            out_path = str(out_dir / file_name)
            result = self.validate_audio(noisy_path, clean_path, out_path, verbose=False)

            for m in metrics_score.keys():
                metrics_score[m] += result[m]

            results.append(result)

        if verbose:
            for key, val in metrics_score.items():
                print(f"{key}: {val / len(files):.3f}")

        with (out_dir / "result.txt").open("w") as f:
            json.dump(results, f, indent=2)
