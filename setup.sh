echo "Install requirements"
pip install -r requirements.txt

echo "Download checkpoint"
gdown "https://drive.google.com/u/0/uc?id=1micEaZlvjWwz-4XCOxn_vbKir03maW9U" -O default_test_model/config.json
gdown "https://drive.google.com/u/0/uc?id=1HBo9uI71b9VVXw1EZCdrf-CF2_D0GL2D" -O default_test_model/model.pth

echo "Download vctk test"
wget "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip"
wget "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip"
unzip -q noisy_testset_wav.zip
rm noisy_testset_wav.zip
unzip -q clean_testset_wav.zip
rm clean_testset_wav.zip