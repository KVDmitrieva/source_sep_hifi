echo "Install requirements"
pip install -q -r requirements.txt
pip install -q gdown

echo "Download checkpoints"
mkdir checkpoints

gdown --fuzzy "https://drive.google.com/file/d/11DEZuCMOLWvTJEHvDqg3I4EBoKVuSpOF/view?usp=drive_link" -O checkpoints.zip
unzip -q checkpoints.zip


echo "Download vctk test"
mkdir data
wget "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip" -P data
wget "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip" -P data
unzip -q data/noisy_testset_wav.zip -d data
rm data/noisy_testset_wav.zip
unzip -q data/clean_testset_wav.zip -d data
rm data/clean_testset_wav.zip
