echo "Install requirements"
pip install -q -r requirements.txt
pip install -q gdown

echo "Download checkpoints"
mkdir checkpoints

gdown --fuzzy "https://drive.google.com/file/d/11DEZuCMOLWvTJEHvDqg3I4EBoKVuSpOF/view?usp=drive_link" -O checkpoints.zip
unzip -q checkpoints.zip

# checkpoints
#  - hifi_official
#  - hifi_plusplus  (150 эпох)
#  - hifi_wo_spec   (150 эпох)
#  - hifi_mrf       (150 эпох)
#  - hifi_fms       (150 эпох)
#  - hifi_mrf_fms   (150 эпох)

#  - hifi_stream           (60 эпох)
#  - hifi_noise_stream     (30 эпох)
#  - hifi_mrf_stream       (80 эпох)
#  - hifi_wo_spec_stream   (80 эпох)
#  - hifi_fms_stream       (70 эпох)
#  - hifi_mrf_fms_stream   (20 эпох)

# - hifi_context           (20 эпох)


echo "Download vctk test"
mkdir data
wget "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip" -P data
wget "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip" -P data
unzip -q data/noisy_testset_wav.zip -d data
rm data/noisy_testset_wav.zip
unzip -q data/clean_testset_wav.zip -d data
rm data/clean_testset_wav.zip
