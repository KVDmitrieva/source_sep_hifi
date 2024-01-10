from enum import Enum

from .tensorboard import TensorboardWriter
from .wandb import WanDBWriter


class VisualizerBackendType(str, Enum):
    tensorboard = "tensorboard"
    wandb = "wandb"


def get_visualizer(config, logger, backend: VisualizerBackendType):
    if backend == VisualizerBackendType.tensorboard:
        return TensorboardWriter(config.log_dir, logger, True)

    if backend == VisualizerBackendType.wandb:
        return WanDBWriter(config, logger)

    return None
