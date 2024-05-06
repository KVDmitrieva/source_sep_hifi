import argparse
import collections
import warnings

import random

import numpy as np
import torch

import src.loss as module_loss
import src.metric as module_metric
import src.model as module_arch
from src.trainer import Trainer, StreamingTrainer, ContextStreamingTrainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def init_model(config, device, prefix="gen"):
    model_type = "generator" if prefix == "gen" else "discriminator"
    model = config.init_obj(config[model_type], module_arch)
    model = model.to(device)

    loss_module = config.init_obj(config[f"{prefix}_loss"], module_loss).to(device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config[f"{prefix}_optimizer"], torch.optim, trainable_params)
    lr_scheduler = config.init_obj(config[f"{prefix}_lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    return model, loss_module, optimizer, lr_scheduler


def main(config, train_mode="base"):
    assert train_mode in ["base", "streaming", "context_streaming"]

    set_seed(config["trainer"].get("random_seed", 123))
    logger = config.get_logger("train")

    dataloaders = get_dataloaders(config)
    device, device_ids = prepare_device(config["n_gpu"])

    generator, gen_loss_module, gen_optimizer, gen_lr_scheduler = init_model(config, device, prefix="gen")
    logger.info(generator)

    discriminator, dis_loss_module, dis_optimizer, dis_lr_scheduler = init_model(config, device, prefix="dis")
    logger.info(discriminator)
    
    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

    metrics = [config.init_obj(metric_dict, module_metric)for metric_dict in config["metrics"]]

    if train_mode == "base":
        trainer_type = Trainer
    elif train_mode == "streaming":
        trainer_type = StreamingTrainer
    else:
        trainer_type = ContextStreamingTrainer

    trainer = trainer_type(
        generator,
        discriminator,
        gen_loss_module,
        dis_loss_module,
        metrics,
        gen_optimizer,
        dis_optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        gen_lr_scheduler=gen_lr_scheduler,
        dis_lr_scheduler=dis_lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")

    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")
    args.add_argument("-t", "--train_mode", default="base", type=str, help="training mode")

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
    ]
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()
    main(config, args.train_mode)
