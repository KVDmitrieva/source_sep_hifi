import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

import torch
import torchaudio

import src.model as module_model
from src.utils import ROOT_PATH
from src.datasets.utils import MelSpectrogram, MelSpectrogramConfig
from src.datasets.streamer import FastFileStreamer
from src.utils.parse_config import ConfigParser
from src.metric import *


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def prepare_metric_log(add_metrics_with_ref: bool = False):
    metric, metric_score = {}, {}
    if torch.cuda.is_available():
        metric["WMOS"] = WMOSMetric()

    if add_metrics_with_ref:
        metric["PESQ"] = PESQMetric()
        metric["SI-SDR"] = SISDRMetric()
        metric["SDR"] = SDRMetric()
        metric["STOI"] = STOIMetric()

    for m in metric.keys():
        metric_score[m] = 0.

    return metric, metric_score


def load_audio(path, target_sr):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)

    return audio_tensor


def main(config, out_dir, noisy_dir, target_dir=None):
    logger = config.get_logger("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = config.init_obj(config["generator"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    target_sr = config["preprocessing"]["sr"]
    mel_spec = MelSpectrogram(MelSpectrogramConfig())
    files = sorted(os.listdir(noisy_dir))

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(exist_ok=True, parents=True)

    noisy_dir, out_dir = Path(noisy_dir), Path(out_dir)
    metric, metric_score = prepare_metric_log(target_dir is not None)
    results = []

    if target_dir is not None:
        target_dir = Path(target_dir)

    for file_name in tqdm(files, desc="Process file"):
        noisy_audio = load_audio(noisy_dir / file_name, target_sr)
        noisy_mel = mel_spec(noisy_audio).to(device)

        gen_audio = model(noisy_mel, noisy_audio.unsqueeze(0).to(device)).squeeze(1)
        path = f"{str(out_dir)}/{file_name}"
        torchaudio.save(path, gen_audio.cpu(), target_sr)

        clean_audio = None if target_dir is None else load_audio(target_dir / file_name, target_sr)
        result = {"file": file_name}
        for m in metric.keys():
            if m == "WMOS":
                result[m] = metric[m](gen_audio)
            else:
                to_pad = clean_audio.shape[1] - gen_audio.shape[1]
                gen_audio = torch.nn.functional.pad(gen_audio, (0, to_pad))
                result[m] = metric[m](gen_audio, clean_audio.to(device)).item()
            metric_score[m] += result[m]

        results.append(result)

    for m in metric_score.keys():
        metric_score[m] /= len(files)

    if len(metric_score) > 0:
        print("Mean score:")
        for key, val in metric_score.items():
            print(key, val)

        metric_score["file"] = "Mean score"
        results.append(metric_score)

        with (out_dir / "result.txt").open("w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file noisy_path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="noisy_path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output",
        type=str,
        help="Dir to write results",
    )
    args.add_argument(
        "-n",
        "--noisy_dir",
        default=None,
        type=str,
        help="Path to noisy audios",
    )
    args.add_argument(
        "-t"
        "--target_dir",
        default=None,
        type=str,
        help="Path to target audios",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config, args.output, args.noisy_dir, args.target_dir)
