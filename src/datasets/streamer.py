import numpy as np
from librosa.util import frame


class FileStreamerError(Exception):
    # Basic class for profiler exceptions
    def __init__(self):
        self.msg = "FileStreamerError, something is wrong..."

    def __str__(self):
        # print it
        return self.msg


class MixTargetNotMatchingError(FileStreamerError):

    # Raised when profiler cannot find a pre-specified node
    def __init__(self, len1, len2):
        """
        Input
            str nodeName -- the name of the ProfilerNode which we failed to find
            str rootName -- the name of the root ProfilerNode where the search was executed

        """
        # Error message
        self.msg = "Mix and Target audio have different lengths: " + str(len1) + " and " + str(len2)


class FileStreamer:
    """
    Makes list of chunks from the audio
    """
    def __init__(self, chunk_size, window_delta):
        """
        Input
        int chunk_size -- the size of the chunks to create
        int window_delta -- the shift of the window made at each (set =chunk_size if you want the chunks to be not intersecting each other)
                            A good choice is also to set = chunk_size//2 to get half-intersecting chunks
        """
        self.chunk_size = chunk_size
        self.window_delta = window_delta

    def __call__(self, s_mix, s_target=None):
        chunks_mix, chunks_target = self.forward(s_mix, s_target)

        return chunks_mix, chunks_target

    def forward(self, signal_mix, signal_target):
        """
        Calls FileStreamer to get list of chunks from two audios (assuming mix and target have the same length)
        Input
        float[] signal_mix -- mixed audio (N,)
        float[] signal_target -- mixed audio (N,), can be None, in this case an empty list is returned
        Output
        two lists of chunks: chunks_mix, chunks_target
        """

        if not (signal_target is None):
            if not (signal_mix.shape[-1] == signal_target.shape[-1]):
                raise MixTargetNotMatchingError(signal_mix.shape[-1], signal_target.shape[-1])

        chunks_mix = []
        chunks_target = []

        audio_len = len(signal_mix)
        current_chunk_start = 0
        while current_chunk_start < audio_len - self.window_delta:
            if self.chunk_size == -1:
                current_chunk_end = audio_len
            else:
                current_chunk_end = min(current_chunk_start + self.chunk_size, audio_len)

            mix_chunk = signal_mix[current_chunk_start:current_chunk_end]
            if signal_target is not None:
                target_chunk = signal_target[current_chunk_start:current_chunk_end]

            if len(mix_chunk) < self.chunk_size:
                to_pad = self.chunk_size - len(mix_chunk)
                mix_chunk = np.pad(mix_chunk, (0, to_pad), 'constant', constant_values=(0, 1e-5))

                if signal_target is not None:
                    target_chunk = np.pad(target_chunk, (0, to_pad), 'constant', constant_values=(0, 1e-5))

            chunks_mix.append(mix_chunk)
            if signal_target is not None:
                chunks_target.append(target_chunk)

            current_chunk_start = current_chunk_start + self.window_delta

        return chunks_mix, chunks_target


class FastFileStreamer(FileStreamer):
    """
    Makes list of chunks from the audio
    """
    def __init__(self, chunk_size, window_delta):
        """
        Input
        int chunk_size -- the size of the chunks to create
        int window_delta -- the shift of the window made at each (set =chunk_size if you want the chunks to be not intersecting each other)
                            A good choice is also to set = chunk_size//2 to get half-intersecting chunks
        """
        super().__init__(chunk_size, window_delta)

    def forward(self, signal_mix, signal_target):
        """
        Calls FileStreamer to get list of chunks from two audios (assuming mix and target have the same length)
        Input
        float[] signal_mix -- mixed audio (N, )
        float[] signal_target -- mixed audio (N, ), can be None, in this case an empty list is returned
        Output
        two lists of chunks: chunks_mix, chunks_target
        """

        if not (signal_target is None):
            if not (signal_mix.shape[-1] == signal_target.shape[-1]):
                raise MixTargetNotMatchingError(signal_mix.shape[-1], signal_target.shape[-1])

        chunks_mix = self._process_audio(signal_mix)
        chunks_target = [] if signal_target is None else self._process_audio(signal_target)

        return chunks_mix, chunks_target

    def _process_audio(self, signal):
        """
        Input
        float[] signal -- audio (time, )
        Output
        float[] chunked_signal -- chunked audio (num_chunks, len_chunk)
        """
        num_mix = signal.shape[-1] - self.chunk_size
        n_mix = num_mix // self.window_delta + (num_mix % self.window_delta > 0)
        n_pad = self.window_delta * n_mix + self.chunk_size - signal.shape[-1]

        signal = np.pad(signal, (0, n_pad), 'constant', constant_values=(0, 1e-5))
        chunks = frame(signal, frame_length=self.chunk_size, hop_length=self.window_delta)
        return chunks.transpose(1, 0)


class FastFileStreamerBatched(FastFileStreamer):
    """
    Makes list of chunks from the audio
    """
    def __init__(self, chunk_size, window_delta):
        """
        Input
        int chunk_size -- the size of the chunks to create
        int window_delta -- the shift of the window made at each (set =chunk_size if you want the chunks to be not intersecting each other)
                            A good choice is also to set = chunk_size//2 to get half-intersecting chunks
        """
        super().__init__(chunk_size, window_delta)

    def _process_audio(self, signal):
        """
        Input
        float[] signal -- audio (batch_size, time)
        Output
        float[] chunked_signal -- chunked audio (batch_size, num_chunks, len_chunk)
        """
        num_mix = signal.shape[-1] - self.chunk_size
        n_mix = num_mix // self.window_delta + (num_mix % self.window_delta > 0)
        n_pad = self.window_delta * n_mix + self.chunk_size - signal.shape[-1]

        signal = np.pad(signal, ((0, 0), (0, n_pad)), 'constant', constant_values=(0, 1e-5))
        chunks = frame(signal, frame_length=self.chunk_size, hop_length=self.window_delta)
        return chunks.transpose(0, 2, 1)
