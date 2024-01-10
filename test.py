import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio

import src.model as module_model
from src.utils import ROOT_PATH
from src.datasets.utils import MelSpectrogram, MelSpectrogramConfig
from src.utils.parse_config import ConfigParser


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_dir, test_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["generator"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(exist_ok=True, parents=True)

    mel_spec = MelSpectrogram(MelSpectrogramConfig())

    for i in range(3):
        path = ROOT_PATH / "test_data" / f"audio_{i + 1}.wav"
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        mel = mel_spec(audio_tensor).to(device)

        gen_audio = model(mel)["generator_audio"].squeeze(1).cpu()

        path = f"{str(out_dir)}/gen_audio_{i + 1}.wav"
        torchaudio.save(path, gen_audio, config["preprocessing"]["sr"])


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
        "--test",
        default=ROOT_PATH / "test_data",
        type=str,
        help="Path to test audios",
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

    main(config, args.output, args.test)
