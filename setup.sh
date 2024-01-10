echo "Install requirements"
pip install -r requirements.txt

echo "Download checkpoint"
gdown "https://drive.google.com/u/0/uc?id=1o_4tceuFzwYgexKVwEDIXhLKxWlwFKj_" -O default_test_model/config.json
gdown "https://drive.google.com/u/0/uc?id=1zFmPTrP4sG2ZulWRGyoQwjFZTgpN5ff_" -O default_test_model/model.pth
