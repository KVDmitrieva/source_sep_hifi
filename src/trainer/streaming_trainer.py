import torch

from src.trainer.trainer import Trainer
from src.utils import MetricTracker
from src.datasets.streamer import FastFileStreamerBatched


class StreamingTrainer(Trainer):
    """
    Trainer class
    """

    def __init__(self, generator, discriminator, gen_criterion, dis_criterion, metrics, gen_optimizer, dis_optimizer,
                 config, device, dataloaders, gen_lr_scheduler=None, dis_lr_scheduler=None, len_epoch=None, skip_oom=True):
        super().__init__(generator, discriminator, gen_criterion, dis_criterion, metrics, gen_optimizer,
                         dis_optimizer, config, device, dataloaders, gen_lr_scheduler, dis_lr_scheduler, len_epoch, skip_oom)
        self.streamer = FastFileStreamerBatched(chunk_size=config["streamer"]["chunk_size"],
                                                window_delta=config["streamer"]["window_delta"])

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["target_audio", "target_mel"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        chunks, _ = self.streamer(batch["audio"].cpu().squeeze(1).numpy())
        bs, chunk_num, chunk_size = chunks.shape

        batch["chunked_audio"] = torch.from_numpy(chunks.reshape(-1, chunks.shape[-1])).to(self.device)
        batch["chunked_mel"] = self.mel_spec(batch["chunked_audio"])
        batch["chunked_audio"] = batch["chunked_audio"].unsqueeze(1)

        batch = self.move_batch_to_device(batch, self.device)

        # generator_audio
        gen_chunks = self.generator(batch["chunked_mel"], batch["chunked_audio"])
        gen_chunks = gen_chunks.reshape(bs, chunk_num, chunk_size)
        batch["generator_audio"] = self.overlap_add_batched(
            gen_chunks, self.streamer.window_delta, self.streamer.chunk_size
        )

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

    @staticmethod
    def overlap_add_batched(chunks, window_delta, chunk_size):
        bs, chunk_num, _ = chunks.shape
        first_elements = chunks[:, 0, 0:window_delta]
        last_elements = chunks[:, -1, window_delta:]

        overlapped_chunks = chunks.reshape((bs, -1))[:, window_delta: chunk_size * chunk_num - window_delta].reshape((bs, -1, chunk_size))
        eye_matrix = torch.eye(window_delta, dtype=overlapped_chunks.dtype, device=chunks.device).repeat((2 , 1))
        sum_overlappes_chunks = (overlapped_chunks @ eye_matrix).reshape(bs, -1)

        audio_without_last_elems = torch.cat((first_elements, sum_overlappes_chunks), dim=-1)
        audio = torch.cat((audio_without_last_elems, last_elements), dim=-1)
        return audio

    @staticmethod
    def overlap_add_sin_batched(chunks, window_delta, chunk_size):
        bs, chunk_num, _ = chunks.shape
        first_elements = chunks[:, 0, 0:window_delta]
        last_elements = chunks[:, -1, window_delta:]

        overlapped_chunks = chunks.reshape((bs, -1))[:, window_delta: chunk_size * chunk_num - window_delta].reshape((bs, -1, chunk_size))
        window_vec = torch.sin(torch.arange(window_delta) / (window_delta - 1)) * (torch.pi / 2)
        kernel_matrix = torch.zeros(2 * window_delta, window_delta, dtype=overlapped_chunks.dtype, device=chunks.device)
        kernel_matrix[:window_delta, :] = torch.diag(1 - window_vec)
        kernel_matrix[window_delta:, :] = torch.diag(window_vec)
        sum_overlappes_chunks = (overlapped_chunks @ kernel_matrix).reshape(bs, -1)

        audio_without_last_elems = torch.cat((first_elements, sum_overlappes_chunks), dim=-1)
        audio = torch.cat((audio_without_last_elems, last_elements), dim=-1)
        return audio

