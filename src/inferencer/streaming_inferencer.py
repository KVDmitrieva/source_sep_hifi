import json
import os
from pathlib import Path

from tqdm import tqdm

import torch
import torchaudio

from src.inferencer.inferencer import Inferencer
from src.datasets.streamer import FastFileStreamer


class StreamingInferencer(Inferencer):
    def __init__(self, config, chunk_size, window_delta, segment_size=None):
        super().__init__(config, segment_size)

        # TODO: get chunk_size & window_delta from config
        self.chunk_size = chunk_size
        self.window_delta = window_delta

        self.streamer = FastFileStreamer(chunk_size, window_delta)

    def denoise_streaming_audio(self, noisy_path: str, out_path: str = "result.wav",
                                mode: str = "overlap_add", normalize_chunk: bool = True):
        assert mode in ["overlap_add", "overlap_add_sin", "overlap_nonintersec"], "invalid overlap mode"

        noisy_audio = self._load_audio(noisy_path)
        noisy_audio = self._cut_audio(noisy_audio, new_ind=True)

        noisy_chunks, _ = self.streamer(noisy_audio.squeeze(0), None)

        outputs = []
        for chunk in noisy_chunks:
            chunk = torch.tensor(chunk).unsqueeze(0)
            mel_chunk = self.mel_spec(chunk).to(self.device)
            with torch.no_grad():
                gen_chunk = self.model(mel_chunk, chunk.unsqueeze(0).to(self.device))

            to_pad = chunk.shape[-1] - gen_chunk.shape[-1]
            gen_chunk = torch.nn.functional.pad(gen_chunk, (0, to_pad))
            # if normalize_chunk:
            #     gen_chunk = gen_chunk / torch.amax(torch.abs(gen_chunk))

            outputs.append(gen_chunk.cpu().squeeze())

        if mode == "overlap_add":
            gen_audio = self.overlap_add(outputs, self.window_delta, self.chunk_size)
        elif mode == "overlap_add_sin":
            gen_audio = self.overlap_add_sin(outputs, self.window_delta, self.chunk_size)
        else:
            gen_audio = self.overlap_nonintersec(outputs, self.window_delta, self.chunk_size)

        gen_audio = gen_audio.unsqueeze(0)
        if out_path is not None:
            torchaudio.save(out_path, gen_audio, self.target_sr)

        return gen_audio

    def denoise_streaming_dir(self, noisy_dir: str, out_dir: str = "output", mode: str = "overlap_add"):
        assert Path(noisy_dir).exists(), "invalid noisy_path"

        if not Path(out_dir).exists():
            Path(out_dir).mkdir(exist_ok=True, parents=True)

        files = sorted(os.listdir(noisy_dir))
        noisy_dir, out_dir = Path(noisy_dir), Path(out_dir)

        for file_name in tqdm(files, desc="Process file"):
            noisy_path = str(noisy_dir / file_name)
            out_path = str(out_dir / file_name)
            _ = self.denoise_streaming_audio(noisy_path, out_path, mode)

    def validate_streaming_audio(self, noisy_path: str, clean_path: str, out_path: str = "result.wav", mode: str = "overlap_add", verbose=True):
        gen_audio = self.denoise_streaming_audio(noisy_path, out_path, mode)
        clean_audio = self._load_audio(clean_path)
        clean_audio = self._cut_audio(clean_audio, new_ind=False)

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

    def validate_streaming_dir(self, noisy_dir: str, clean_dir: str, out_dir: str = "output", mode: str = "overlap_add", verbose=True):
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
            result = self.validate_streaming_audio(noisy_path, clean_path, out_path, mode, verbose=False)

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
    def overlap_add(chunks, window_delta, chunk_size):
        res = torch.zeros(window_delta * len(chunks) + chunk_size)
        for (i, ch) in enumerate(chunks):
            res[window_delta * i:window_delta * i + chunk_size] += ch

        return res

    @staticmethod
    def overlap_add_sin(chunks, window_delta, chunk_size):
        window = torch.sin((torch.arange(window_delta) / (window_delta - 1)) * (torch.pi / 2))
        res = torch.zeros(window_delta * len(chunks) + chunk_size)
        for (i, ch) in enumerate(chunks):
            if i == 0:
                res[:chunk_size] = ch
            else:
                overlap = ch[:chunk_size - window_delta] * window + res[window_delta * i:window_delta * (
                            i - 1) + chunk_size] * (1 - window)
                res[window_delta * i:window_delta * (i - 1) + chunk_size] = overlap
                res[window_delta * (i - 1) + chunk_size:window_delta * i + chunk_size] = ch[chunk_size - window_delta:]

        return res

    @staticmethod
    def overlap_nonintersec(chunks, window_delta, chunk_size):
        res = torch.tensor([])
        for (i, ch) in enumerate(chunks):
            if i == 0:
                res = torch.cat([res, ch], dim=-1)
            else:
                res = torch.cat([res, ch[-window_delta:]], dim=-1)
        return res
