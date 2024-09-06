import torch
import torchaudio

from src.inferencer.inferencer import Inferencer
from src.datasets.streamer import FastFileStreamer


class StreamingInferencer(Inferencer):
    def __init__(self, config, chunk_size : int, window_delta : int, mode: str = "overlap_add", normalize_chunk=False, segment_size: int = None):
        assert mode in ["overlap_add", "overlap_add_sin", "overlap_nonintersec"], "invalid overlap mode"
        super().__init__(config, segment_size)

        self.chunk_size = chunk_size
        self.window_delta = window_delta
        self.normalize_chunk = normalize_chunk

        if mode == "overlap_add":
            self.overlap = self.overlap_add
        elif mode == "overlap_add_sin":
            self.overlap = self.overlap_add_sin
        else:
            self.overlap = self.overlap_nonintersec

        self.streamer = FastFileStreamer(chunk_size, window_delta)

    def denoise_audio(self, noisy_path: str, out_path: str = "result.wav"):
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
            outputs.append(gen_chunk.cpu().squeeze())

        gen_audio = self.overlap(outputs, self.window_delta, self.chunk_size)
        gen_audio = gen_audio.unsqueeze(0)
        
        if out_path is not None:
            torchaudio.save(out_path, gen_audio, self.target_sr)

        return gen_audio

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
                overlap = ch[:chunk_size - window_delta] * window + res[window_delta * i:window_delta * (i - 1) + chunk_size] * (1 - window)
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
