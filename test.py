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
from src.utils.parse_config import ConfigParser
from src.metric import *


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_dir, test_dir, target_dir=None):
    logger = config.get_logger("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_sr = config["preprocessing"]["sr"]

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

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(exist_ok=True, parents=True)

    mel_spec = MelSpectrogram(MelSpectrogramConfig())

    files = sorted(os.listdir(test_dir))

    test_dir = Path(test_dir)
    out_dir = Path(out_dir)
    results = []
    metric = {}
    metric_score = {}
    if torch.cuda.is_available():
        metric["WMOS"] = WMOSMetric()
        metric_score["WMOS"] = 0.

    if target_dir is not None:
        target_dir = Path(target_dir)

        metric["PESQ"] = PESQMetric()
        metric["SI-SDR"] = SISDRMetric()
        metric["SDR"] = SDRMetric()
        metric["STOI"] = STOIMetric()

        metric_score["PESQ"] = 0.
        metric_score["SI-SDR"] = 0.
        metric_score["SDR"] = 0.
        metric_score["STOI"] = 0.

    for f in tqdm(files, desc="Process file"):
        path = test_dir / f
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        mel = mel_spec(audio_tensor).to(device)

        gen_audio = model(mel, audio_tensor.unsqueeze(0).to(device)).squeeze(1)
        path = f"{str(out_dir)}/{f}"
        torchaudio.save(path, gen_audio.cpu(), config["preprocessing"]["sr"])

        if len(metric.keys()) > 0:
            result = {"file": f}
            if target_dir is not None:
                path = target_dir / f
                audio_tensor, sr = torchaudio.load(path)
                audio_tensor = audio_tensor[0:1, :]
                if sr != target_sr:
                    audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)

            for m in metric.keys():
                if m == "WMOS":
                    result[m] = metric[m](gen_audio)
                else:
                    result[m] = metric[m](gen_audio, audio_tensor[:, :gen_audio.shape[1]].to(device)).item()
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
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
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
        "-t",
        "--test_dir",
        default=None,
        type=str,
        help="Path to test audios",
    )
    args.add_argument(
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

    main(config, args.output, args.test_dir, args.target_dir)
