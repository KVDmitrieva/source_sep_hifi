from abc import abstractmethod

import torch
from numpy import inf

from src.model.base_model import BaseModel
from src.logger import get_visualizer


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, generator: BaseModel, discriminator: BaseModel, gen_criterion, dis_criterion, metrics,
                 gen_optimizer, dis_optimizer, gen_lr_scheduler, dis_lr_scheduler, config, device):
        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.generator = generator
        self.discriminator = discriminator
        self.gen_criterion = gen_criterion
        self.dis_criterion = dis_criterion
        self.metrics = metrics
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.gen_lr_scheduler = gen_lr_scheduler
        self.dis_lr_scheduler = dis_lr_scheduler

        # for interrupt saving
        self._last_epoch = 0

        self.epochs = config.trainer.epochs
        self.save_period = config.trainer.save_period
        self.monitor = config.trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = config.trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = get_visualizer(
            config, self.logger, config.trainer.visualize
        )

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self.mnt_mode == "min":
                        improved = log[self.mnt_metric] <= self.mnt_best
                    elif self.mnt_mode == "max":
                        improved = log[self.mnt_metric] >= self.mnt_best
                    else:
                        improved = False
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, model_type="gen", save_best=best, only_best=True)
                self._save_checkpoint(epoch, model_type="dis", save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, model_type, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model = self.generator if model_type == "gen" else self.discriminator
        optimizer = self.gen_optimizer if model_type == "gen" else self.dis_optimizer
        lr_scheduler = self.gen_lr_scheduler if model_type == "gen" else self.dis_lr_scheduler

        arch = type(model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "{}_checkpoint-epoch{}.pth".format(model_type, epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / f"{model_type}_model_best.pth")
            torch.save(state, best_path)
            self.logger.info(f"Saving current best: {model_type}_model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        gen_resume_path = str(resume_path)

        dis_resume_path = '/'.join(gen_resume_path.split('/')[:-1]) + "/discriminator.pth"

        self.logger.info("Loading generator checkpoint: {} ...".format(gen_resume_path))
        gen_checkpoint = torch.load(gen_resume_path, self.device)

        self.logger.info("Loading discriminator checkpoint: {} ...".format(dis_resume_path))
        dis_checkpoint = torch.load(dis_resume_path, self.device)

        self.start_epoch = gen_checkpoint["epoch"] + 1
        self.mnt_best = gen_checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if gen_checkpoint["config"]["generator"] != self.config["generator"] or gen_checkpoint["config"]["discriminator"] != self.config["discriminator"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.generator.load_state_dict(gen_checkpoint["state_dict"])
        self.discriminator.load_state_dict(dis_checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                gen_checkpoint["config"]["gen_optimizer"] != self.config["gen_optimizer"] or
                gen_checkpoint["config"]["gen_lr_scheduler"] != self.config["gen_lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.gen_optimizer.load_state_dict(gen_checkpoint["optimizer"])
            self.dis_optimizer.load_state_dict(dis_checkpoint["optimizer"])
            self.gen_lr_scheduler.load_state_dict(gen_checkpoint["lr_scheduler"])
            self.dis_lr_scheduler.load_state_dict(dis_checkpoint["lr_scheduler"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
