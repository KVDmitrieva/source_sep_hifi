# Speech Enhancement project

## Installation guide
Run
```shell
pip install -r requirements.txt
```
to install all libs.

## Train running guide
All configs can be found in the `src/configs` directory. 
Training can be launched with the following command:
```shell
python3 train.py -c src/configs/CONFIG_NAME.json 
```

In order to resume training from checkpoint, use

```shell
python3 train.py -c src/configs/CONFIG_NAME.json -r PATH/TO/CHECKPOINT/generator.pth
```
note that `PATH/TO/CHECKPOINT/` directory should contain discriminator checkpoint (`discriminator.pth`) and `config.json`


## Test running guide
Download all available checkpoints and test data with `setup.sh`:
```shell
./setup.sh
```
Run test with:

```shell
python3 test.py -c MODEL_CONFIG.json \
                 -r MODEL_CHECKPOINT.pth \
                 -o DIRECTORY_FOR_ENHANCED_AUDIO \
                 -n DIRECTORY_WITH_NOISY_AUDIO_TO_ENHANCE \
                 -t DIRECTORY_WITH_CLEAN_AUDIO_TO_VALIDATE_RESULT # optional
```
Example for `hifi_plus` checkpoint:
```shell
python3 test.py -c checkponts/hifi_plusplus/config.json \
                -r checkponts/hifi_plusplus/generator.pth \
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