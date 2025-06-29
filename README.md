# HiFi-Stream: Streaming Speech Enhancement with Generative Adversarial Networks


This is an official implementation of paper [HiFi-Stream: Streaming Speech Enhancement with Generative Adversarial Networks](https://arxiv.org/pdf/2503.17141).

## Installation guide
Run
```shell
git clone https://github.com/KVDmitrieva/source_sep_hifi
cd source_sep_hifi
python -m venv hifi_stream_env
. hifi_stream_env/bin/activate
pip install -r requirements.txt
```
to install all libs

## Training guide

### Configuration
All configuration files are located in `src/configs` directory

### Running Training
Training can be launched with the following command:
```shell
python3 train.py -c src/configs/CONFIG_NAME.json 
```
**Note**: Update the data paths in the relevant config file (`CONFIG_NAME.json`) before running


### Resuming from a Checkpoint
To resume training from a saved checkpoint, run:

```shell
python3 train.py -c src/configs/CONFIG_NAME.json -r PATH/TO/CHECKPOINT/generator.pth
```
**requirements for resuming:**

The checkpoint directory (`PATH/TO/CHECKPOINT/`) must include:
- Generator weights: `generator.pth`
- Discriminator weights: `discriminator.pth`
- Config file: `config.json`


## Test running guide
Download all available checkpoints and test data with `setup.sh`:
```shell
./setup.sh
```
Inference can be launched with the following command:

```shell
python3 test.py -c MODEL_CONFIG.json \
                 -r MODEL_CHECKPOINT.pth \
                 -o DIRECTORY_FOR_ENHANCED_AUDIO \
                 -n DIRECTORY_WITH_NOISY_AUDIO_TO_ENHANCE \
                 -t DIRECTORY_WITH_CLEAN_AUDIO_TO_VALIDATE_RESULT # optional
```
Example for `hifi_plus` checkpoint:
```shell
python3 test.py -c checkpoints/hifi_plusplus/config.json \
                -r checkpoints/hifi_plusplus/generator.pth \
                -o hifi_plusplus_out \
                -n data/noisy_testset_wav \
                -t data/clean_testset_wav
```

To run `streaming_test.py` you should provide overlap mode (overlap_add or overlap_add_sin):
```shell
python3 streaming_test.py -c checkpoints/hifi_plusplus/config.json \
                            -r checkpoints/hifi_plusplus/generator.pth \
                            -o hifi_plusplus_out \
                            -n data/noisy_testset_wav \
                            -t data/clean_testset_wav \
                            -m "overlap_add"
```

# Config directory structure
```
├── ablation                     # Configs for additional experiments
├── one_batch                    # One-batch debug configs
├── hifi_plus.json               # Config for HiFi++ (reprod.) model
├── hifi_plus_wo_spec.json       # Config for HiFi w/o spec model
├── hifi_2d_mrf.json             # Config for HiFi-2dMRF model
├── hifi_fms.json                # Config for HiFi-Stream model
└── hifi_2d_mrf_fms.json         # Config for HiFi-Stream2D model
```