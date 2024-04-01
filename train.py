import warnings

import numpy as np
import torch

import random

import hydra
from hydra.utils import instantiate

from src.trainer import Trainer
from src.utils.data_utils import get_dataloaders
from src.utils.init_utils import setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="src/hydra_configs", config_name="one_batch_model")
def main(config):
    set_random_seed(config.trainer.random_seed)

    logger = setup_saving_and_logging(config)

    dataloaders = get_dataloaders(config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    generator = instantiate(config.generator).to(device)
    logger.info(generator)
    discriminator = instantiate(config.discriminator).to(device)
    logger.info(discriminator)

    if torch.cuda.device_count() >= config.trainer.n_gpu > 1:
        device_ids = list(range(config.trainer.n_gpu))
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

    gen_loss_module = instantiate(config.gen_loss).to(device)
    dis_loss_module = instantiate(config.dis_loss).to(device)
    metrics = instantiate(config.metrics)

    trainable_params = filter(lambda p: p.requires_grad, generator.parameters())
    gen_optimizer = instantiate(config.gen_optimizer, params=trainable_params)
    gen_lr_scheduler = instantiate(config.gen_lr_scheduler, optimizer=gen_optimizer)

    trainable_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    dis_optimizer = instantiate(config.dis_optimizer, params=trainable_params)
    dis_lr_scheduler = instantiate(config.dis_lr_scheduler, optimizer=dis_optimizer)

    trainer = Trainer(
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
        len_epoch=config.trainer.get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    main()
