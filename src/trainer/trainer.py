import random

import PIL
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.trainer.base_trainer import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker
from src.datasets.utils import MelSpectrogram, MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, generator, discriminator, gen_criterion, dis_criterion, metrics, gen_optimizer, dis_optimizer,
                 config, device, dataloaders, gen_lr_scheduler=None, dis_lr_scheduler=None, len_epoch=None, skip_oom=True):
        super().__init__(generator, discriminator, gen_criterion, dis_criterion, metrics, gen_optimizer,
                         dis_optimizer, gen_lr_scheduler, dis_lr_scheduler, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self._setup_loaders(dataloaders, len_epoch)

        self.log_step = config["trainer"].get("log_step", 50)
        self.log_media = config["trainer"].get("log_media", 1)

        self.train_metrics = MetricTracker(
            "discriminator_loss", "generator_loss", "adv_loss", "mel_loss", "feature_loss",
            "gen grad norm", "dis grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "discriminator_loss", "generator_loss", "adv_loss", "mel_loss", "feature_loss",
            *[m.name for m in self.metrics], writer=self.writer
        )

        self.mel_spec = MelSpectrogram(MelSpectrogramConfig())
        self.mel_spec = self.mel_spec.to(self.device)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        self.discriminator.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="train", total=self.len_epoch)):
            try:
                batch = self.process_batch(batch, is_train=True, metrics=self.train_metrics)
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self._free_memory()
                    continue
                else:
                    raise e
            self.train_metrics.update("gen grad norm", self.get_grad_norm())
            self.train_metrics.update("dis grad norm", self.get_grad_norm(model_type="dis"))

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} GenLoss: {:.6f}  DisLoss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["generator_loss"].item(),
                        batch["discriminator_loss"].item()
                    )
                )
                self.writer.add_scalar("gen learning rate", self.gen_lr_scheduler.get_last_lr()[0])
                self.writer.add_scalar("dis learning rate", self.dis_lr_scheduler.get_last_lr()[0])
                self._log_scalars(self.train_metrics)

                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        if self.dis_lr_scheduler is not None:
            self.dis_lr_scheduler.step()

        if self.gen_lr_scheduler is not None:
            self.gen_lr_scheduler.step()

        if epoch % self.log_media == 0:
            self._log_triplet_audio(batch)
            self._log_triplet_spectrogram(batch)

            for part, dataloader in self.evaluation_dataloaders.items():
                val_log = self._evaluation_epoch(epoch, part, dataloader)
                log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        # generator_audio
        batch["generator_audio"] = self.generator(**batch)

        if is_train:
            self.dis_optimizer.zero_grad()

        # real_discriminator_out, real_feature_map
        batch.update(self.discriminator(batch["target_audio"]))

        # gen_discriminator_out, gen_feature_map
        batch.update(self.discriminator(batch["generator_audio"].detach(), prefix="gen"))

        dis_loss = self.dis_criterion(**batch)
        batch.update(dis_loss)

        if is_train:
            batch["discriminator_loss"].backward()
            self._clip_grad_norm(model_type="dis")
            self.dis_optimizer.step()

        if is_train:
            self.gen_optimizer.zero_grad()

        batch.update(self.discriminator(batch["target_audio"]))
        batch.update(self.discriminator(batch["generator_audio"], prefix="gen"))

        batch["gen_mel"] = self.mel_spec(batch["generator_audio"]).squeeze(1)
        gen_loss = self.gen_criterion(**batch)
        batch.update(gen_loss)

        if is_train:
            gen_loss["generator_loss"].backward()
            self._clip_grad_norm(model_type="gen")
            self.gen_optimizer.step()

        for key in dis_loss.keys():
            metrics.update(key, batch[key].item())
        for key in gen_loss.keys():
            metrics.update(key, batch[key].item())

        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.generator.eval()
        self.discriminator.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.process_batch(batch, is_train=False, metrics=self.evaluation_metrics)
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_triplet_audio(batch)
            self._log_triplet_spectrogram(batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.generator.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        for name, p in self.discriminator.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_grad_norm(self, model_type="gen", norm_type=2):
        model = self.generator if model_type == "gen" else self.discriminator
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        parameters = [p for p in parameters if p.grad is not None]
        parameters_stack = torch.stack([torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters])
        return torch.norm(parameters_stack, norm_type).item()
    
    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["mel", "audio", "target_audio", "target_mel"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, model_type="gen"):
        model = self.generator if model_type == "gen" else self.discriminator
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(model.parameters(), self.config["trainer"]["grad_norm_clip"])
    
    def _free_memory(self):
        self.logger.warning("OOM on batch. Skipping batch.")
        for p in self.generator.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        for p in self.discriminator.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        torch.cuda.empty_cache()

    def _setup_loaders(self, dataloaders, len_epoch):
        use_inf_loop = len_epoch is not None
        self.train_dataloader = inf_loop(dataloaders["train"]) if use_inf_loop else dataloaders["train"]
        self.len_epoch = len_epoch if use_inf_loop else len(self.train_dataloader)
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _log_spectrogram(self, spectrogram_batch, name="spectrogram"):
        spectrogram = random.choice(spectrogram_batch.detach().cpu()).squeeze(0)
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image(name, ToTensor()(image))

    def _log_triplet_spectrogram(self, batch):
        ind = random.randint(0, batch["mel"].shape[0] - 1)
        for name, spectrogram in zip(["noisy_mel", "generated_mel", "target_mel"],  ["mel", "gen_mel", "target_mel"]):
            image = PIL.Image.open(plot_spectrogram_to_buf(batch[spectrogram][ind].detach().cpu()))
            self.writer.add_image(name, ToTensor()(image))

    def _log_triplet_audio(self, batch):
        ind = random.randint(0, batch["audio"].shape[0] - 1)
        self.writer.add_audio("noisy_audio", batch["audio"][ind].detach().cpu(), self.config["preprocessing"]["sr"])
        self.writer.add_audio("generated_audio", batch["generator_audio"][ind].detach().cpu(), self.config["preprocessing"]["sr"])
        self.writer.add_audio("target_audio", batch["target_audio"][ind].detach().detach().cpu(), self.config["preprocessing"]["sr"])

    def _log_audio(self, audio_batch, name="audio"):
        audio = random.choice(audio_batch.detach().cpu())
        self.writer.add_audio(name, audio, self.config["preprocessing"]["sr"])
