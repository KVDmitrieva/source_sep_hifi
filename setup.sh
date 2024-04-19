echo "Install requirements"
pip install -r requirements.txt

echo "Download checkpoints"
mkdir checkpoints
#gdown --folder "https://drive.google.com/drive/folders/1KGnHP5_f7pwelusGoztjpvjyF9LWQ3gL" -O checkpoints/hifi_plusplus  # 80 эпох
#gdown --folder "https://drive.google.com/drive/folders/1pBUU2Rk-cGOYewrS491hS7nLzsThk5hA" -O checkpoints/hifi_noise     # 80 эпох
#gdown --folder "https://drive.google.com/drive/folders/1ZuEA0TG8rIiEn-8_nmAa8ddR-KrpnJlf" -O checkpoints/hifi_mrf       # 80 эпох
#gdown --folder "https://drive.google.com/drive/folders/1NQfny6T2jH0gMUvqFCuL8CuVO0mRii5m" -O checkpoints/hifi_wo_spec   # 80 эпох
#gdown --folder "https://drive.google.com/drive/folders/16xG39ZUC4qSJZaRMF-GOyHubQH6V8E26" -O checkpoints/hifi_fms       # 90 эпох
#gdown --folder "https://drive.google.com/drive/folders/1aUurEr9h1T0tk43VyWi6qdamBnQDHi2Z" -O checkpoints/hifi_fms_noise # 45 эпох
#
#gdown --folder "https://drive.google.com/drive/folders/1KGnHP5_f7pwelusGoztjpvjyF9LWQ3gL" -O checkpoints/hifi_stream           # 50 эпох
#gdown --folder "https://drive.google.com/drive/folders/1pBUU2Rk-cGOYewrS491hS7nLzsThk5hA" -O checkpoints/hifi_noise_stream     # 30 эпох
#gdown --folder "https://drive.google.com/drive/folders/1ZuEA0TG8rIiEn-8_nmAa8ddR-KrpnJlf" -O checkpoints/hifi_mrf_stream       # 60 эпох
#gdown --folder "https://drive.google.com/drive/folders/1NQfny6T2jH0gMUvqFCuL8CuVO0mRii5m" -O checkpoints/hifi_wo_spec_stream   # 40 эпох
#gdown --folder "https://drive.google.com/drive/folders/16xG39ZUC4qSJZaRMF-GOyHubQH6V8E26" -O checkpoints/hifi_fms_stream       # 70 эпох

gdown --fuzzy "https://drive.google.com/file/d/1BPEnaTEEyxpK2w-yNlSt6fJgl2l0qNiJ/view?usp=drive_link" -O checkpoints.zip
unzip -q checkpoints.zip

#mkdir checkpoints/hifi_official
#wget "https://github.com/SamsungLabs/hifi_plusplus/releases/download/w/se.pth" -O checkpoints/hifi_official/generator.pth
#cp "src/configs/hifi_official.json" "checkpoints/hifi_official/config.json"  discriminator


echo "Download vctk test"
mkdir data
wget "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip" -P data
wget "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip" -P data
unzip -q data/noisy_testset_wav.zip -d data
rm data/noisy_testset_wav.zip
unzip -q data/clean_testset_wav.zip -d data
rm data/clean_testset_wav.zip