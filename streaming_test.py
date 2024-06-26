import argparse
import json
import os
from pathlib import Path

from src.utils import ROOT_PATH
from src.utils.parse_config import ConfigParser

from src.inferencer.streaming_inferencer import StreamingInferencer


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")

    args.add_argument("-c", "--config", default=None, type=str, help="config file noisy_path (default: None)",)
    args.add_argument("-r", "--resume", default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()), type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)",)
    args.add_argument("-o", "--output", default="output", type=str, help="Dir to write results")
    args.add_argument("-n", "--noisy_dir", default=None, type=str, help="Path to noisy audios")
    args.add_argument("-t", "--target_dir", default=None, type=str, help="Path to target audios")
    args.add_argument("-j", "--json_path", default=None, type=str, help="Path to json with audio paths")
    args.add_argument("-m", "--overlap_mode", default=None, type=str, help="Overlap mode",)
    args.add_argument("-s", "--chunk_size", default=4096, type=int, help="Chunk size",)
    args.add_argument("--segment_size", default=None, type=int, help="Max audio len")

    args = args.parse_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    inferencer = StreamingInferencer(config, chunk_size=args.chunk_size,
                                     window_delta=args.chunk_size // 2,
                                     segment_size=args.segment_size,
                                     mode=args.overlap_mode)
    
    if args.json_path is not None:
        inferencer.validate_json(args.json_path, args.output)
    elif args.target_dir is None:
        inferencer.denoise_dir(args.noisy_dir, args.output)
    else:
        inferencer.validate_dir(args.noisy_dir, args.target_dir, args.output)
