import json
import os
from pathlib import Path

from tqdm import tqdm

import torch
import torchaudio

import src.model as module_model
from src.datasets.utils import MelSpectrogram, MelSpectrogramConfig
from src.metric import *


class Inferencer:
    def __init__(self, config):
        self.logger = config.get_logger("test")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = config.init_obj(config["generator"], module_model)
        self.logger.info(model)

        self.logger.info("Loading checkpoint: {} ...".format(config.resume))
        checkpoint = torch.load(config.resume, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        if config["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        self.model = model.to(self.device)
        self.model.eval()

        self.mel_spec = MelSpectrogram(MelSpectrogramConfig())
        self.target_sr = config["preprocessing"]["sr"]
        self.metrics = self._prepare_metrics()

    @staticmethod
    def _prepare_metrics():
        metric = {}
        if torch.cuda.is_available():
            metric["WMOS"] = WMOSMetric()

        metric["PESQ"] = PESQMetric()
        metric["SI-SDR"] = SISDRMetric()
        metric["SDR"] = SDRMetric()
        metric["STOI"] = STOIMetric()

        return metric

    def _load_audio(self, path: str):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.target_sr)

        return audio_tensor

    def denoise_audio(self, noisy_path: str, out_path: str = "result.wav"):
        noisy_audio = self._load_audio(noisy_path)
        noisy_mel = self.mel_spec(noisy_audio).to(self.device)

        gen_audio = self.model(noisy_mel, noisy_audio.unsqueeze(0).to(self.device))
        gen_audio = gen_audio.cpu().squeeze(1)
        if out_path is not None:
            torchaudio.save(out_path, gen_audio, self.target_sr)

        return gen_audio

    def denoise_dir(self, noisy_dir: str, out_dir: str = "output"):
        assert Path(noisy_dir).exists(), "invalid noisy_path"

        if not Path(out_dir).exists():
            Path(out_dir).mkdir(exist_ok=True, parents=True)

        files = sorted(os.listdir(noisy_dir))
        noisy_dir, out_dir = Path(noisy_dir), Path(out_dir)

        for file_name in tqdm(files, desc="Process file"):
            noisy_path = str(noisy_dir / file_name)
            out_path = str(out_dir / file_name)
            _ = self.denoise_audio(noisy_path, out_path)

    def validate_audio(self, noisy_path: str, clean_path: str, out_path: str = "result.wav", verbose=True):
        gen_audio = self.denoise_audio(noisy_path, out_path)
        clean_audio = self._load_audio(clean_path)

        result = {}
        for m in self.metrics.keys():
            if m == "WMOS":
                result[m] = self.metrics[m](gen_audio.to(self.device))
            else:
                to_pad = clean_audio.shape[1] - gen_audio.shape[1]
                gen_audio = torch.nn.functional.pad(gen_audio, (0, to_pad))
                result[m] = self.metric[m](gen_audio, clean_audio).item()

            if verbose:
                print(f"{m}: {result[m]:.3f}")

        return result

    def validate_dir(self, noisy_dir: str, clean_dir: str, out_dir: str = "output", verbose=True):
        assert Path(noisy_dir).exists(), "invalid noisy dir"
        assert Path(clean_dir).exists(), "invalid clean dir"

        if not Path(out_dir).exists():
            Path(out_dir).mkdir(exist_ok=True, parents=True)

        files = sorted(os.listdir(noisy_dir))
        noisy_dir, clean_dir, out_dir = Path(noisy_dir), Path(clean_dir), Path(out_dir)

        results = []
        metrics_score = {}
        for m in self.metrics.keys():
            metrics_score[m] = 0.

        for file_name in tqdm(files, desc="Process file"):
            noisy_path = str(noisy_dir / file_name)
            clean_path = str(clean_dir / file_name)
            out_path = str(out_dir / file_name)
            result = self.validate_audio(noisy_path, clean_path, out_path, verbose=False)

            for m in metrics_score.keys():
                metrics_score[m] += result[m]

            result[file_name] = result
            results.append(result)

        if verbose:
            for key, val in metrics_score.items():
                print(f"{key}: {val / len(files):.3f}")

        with (out_dir / "result.txt").open("w") as f:
            json.dump(results, f, indent=2)










class StreamingInferencer_2(object):
    '''
    Inferencer base class: object used to run simple and validation inference
    '''

    def __init__(self, model, config, useProfiler, dir):
        '''
        Input
        TorchModel model -- taken from hdiSpSep.models
        LoadConfig config -- config (in hdiSpSep.utils.generic_utils.load_config config parser)
        bool useProfiler -- whether to use torch.profiler
        str dir -- the noisy_path to the dataset for DatasetReader (in case you are planning to use validationRun)
        '''

        self.config = config

        #logging
        self.inferencerLogPath = config.inferencerLog_path
        self.logger = get_logger(os.path.join(self.inferencerLogPath, "inferencer.log"), file=False)
        self.logging_period = config.logging_period
        self.reporter = InferenceReporter(self.logger, period=self.logging_period)#re-define in children classes if needed

        self.useProfiler=useProfiler

        #choose the device
        if(torch.cuda.is_available()):
            self.logger.info("CUDA is available, using GPU for computations")
        else:
            self.logger.info("CUDA is unavailable, using CPU for computations")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device) #send the model there

        self.checkpointPath = config.checkpoint_path
        self.logger.info("Using checkpoint: %s" % self.checkpointPath)

        #set up the checkpoint
        checkpoint = torch.load(self.checkpointPath, map_location='cpu')

        try:
            self.model.load_state_dict(checkpoint['model'])
        except:
            self.logger.info("WARNING! load_state_dict_failed, expected load_state_dict in the child init")

        self.model.eval()

        self.audioProcessor = AudioProcessor(config.audio)
        self.dataReader = None
        if dir:
            self.dataReader = DatasetReader(config, self.audioProcessor, dir)


    def process_audio(self, audio):
        audio_processed = np.where((audio < 0) & (audio >= -0.001), -0.001, audio)
        audio_processed = np.where((audio_processed > 0) & (audio_processed <= 0.001), 0.001, audio_processed)

        return audio_processed


    def simpleRun(self, mix, ref):
        '''
        Just runs the models and returns the result of the extraction
        Input
        mix -- np.array [Nmix] -- mixed signal
        ref -- np.array [Nref] -- reference signal
        '''
        raise NotImplementedError

    def simpleRunStreamed(self, mix, ref):
        '''
        Just runs the models and returns the result of the extraction (streaming processing, treating ref and target as threads)
        Input
        mix -- np.array [Nmix] -- mixed signal
        ref -- np.array [Nref] -- reference signal
        '''
        raise NotImplementedError

    def overlap_add(self, chunks, windowDelta, chunkSize):
        res = np.zeros(windowDelta * len(chunks) + chunkSize)
        for (i, ch) in enumerate(chunks):
            res[windowDelta*i:windowDelta*i+chunkSize] += ch

        return res

    def overlap_add_sin(self, chunks, windowDelta, chunkSize):
        import matplotlib.pyplot as plt

        window = np.sin((np.arange(windowDelta) / (windowDelta - 1)) * (np.pi / 2))
        # window = np.ones(windowDelta)
        res = np.zeros(windowDelta * len(chunks) + chunkSize)
        for (i, ch) in enumerate(chunks):
            if i == 0:
                res[:chunkSize] = ch
            else:
                overlap = ch[:chunkSize-windowDelta]*window+res[windowDelta*i:windowDelta*(i-1)+chunkSize] * (1 - window)
                res[windowDelta*i:windowDelta*(i-1)+chunkSize] = overlap
                res[windowDelta * (i-1)+chunkSize:windowDelta*i+chunkSize] = ch[chunkSize-windowDelta:]

        return res

    def overlap_nonintersec(self, chunks, windowDelta, chunkSize):
        res = np.array([])
        for (i, ch) in enumerate(chunks):
            if i == 0:
                res = np.append(res, ch)
            else:
                res = np.append(res, ch[-windowDelta:])
        return res


    def validationRun(self, outPath=None, streamed=False,  normalizeChunk=True, normalizeResult=False, preCompRef=False, finalAssemblyMethod="overlap_add", batched=False):
        '''
        Runs a session of validation inference
        Input
        str outPath -- noisy_path to the log dump
        '''

        cur_iter = 0
        num_iters = len(self.dataReader)
        time0Overall = time.time()
        update_steps = 1000

        #for clear reading
        def profilerIf(func, mix, ref, **run_params):
            if(self.useProfiler):
                with profiler.record_function(str(func.__qualname__)+" in Inferencer.validationRun"):
                    predict = func(mix, ref, **run_params)
            else:
                predict = func(mix, ref, **run_params)
            return predict

        preCompRef = streamed and preCompRef
        for egs in self.dataReader:
            key, ref, mix, target = egs

            time0Inference = time.time()#inference time measurement

            if(streamed):
                predict = profilerIf(self.simpleRunStreamed, mix, ref, normalizeChunk=normalizeChunk, normalizeResult=normalizeResult,\
                                      preCompRef=preCompRef, finalAssemblyMethod=finalAssemblyMethod, batched=batched)
            else:
                predict = profilerIf(self.simpleRun, mix, ref, normalizeResult=normalizeResult, preCompRef=preCompRef)

            cur_iter += 1

            if len(predict) < len(target):
                warnings.warn(f"Target length is {len(target)}, Predict length is {len(predict)}")
                l = len(target) - len(predict)
                predict = np.pad(predict, (0, l), 'constant', constant_values=(0, 0))
            else:
                predict = predict[:len(target)]

            predict_processed = self.process_audio(predict)
            target_processed = self.process_audio(target)

            try:
                self.reporter.add(*calcMetrics(predict_processed, target_processed), \
                                key=key, mixLen=mix.shape[0], refLen=ref.shape[0],\
                                sr=self.config.audio["voicefilter"]["sample_rate"],\
                                inferenceTime=time.time()-time0Inference)
            except ValueError:
                self.logger.info(f"Error on file: {key}")

            if cur_iter % (max(num_iters // update_steps, 1)) == 0:
                durationOverall = time.time() - time0Overall
                self.logger.info(f"Files Processed | {cur_iter} out of {num_iters}; {(cur_iter / num_iters) * 100:.2f}% in time {durationOverall}")
                time0Overall = time.time()

        self.reporter.report()
        if outPath:
            self.reporter.saveInferencerData(outPath, self.dataReader.dataset_dir, self.config.model_name, self.config.checkpoint_path)

        self.reporter.reset()

