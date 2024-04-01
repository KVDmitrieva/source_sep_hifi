from typing import List

from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    pad_value = -11.5129251
    spectrogram, audio = [], []
    target_spec, target_audio = [], []

    for item in dataset_items:
        audio.append(item["audio"].T)
        target_audio.append(item["target_audio"].T)
        spectrogram.append(item["spectrogram"].squeeze(0).T)
        target_spec.append(item["target_spectrogram"].squeeze(0).T)

    return {
        "audio": pad_sequence(audio, batch_first=True).transpose(1, 2),
        "target_audio": pad_sequence(target_audio, batch_first=True).transpose(1, 2),
        "mel": pad_sequence(spectrogram, batch_first=True, padding_value=pad_value).transpose(1, 2),
        "target_mel": pad_sequence(target_spec, batch_first=True, padding_value=pad_value).transpose(1, 2)
    }
