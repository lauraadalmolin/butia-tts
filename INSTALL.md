
Mozilla TTS Install and Setup for this Project
=========================
In order to use this repository, run this code first.
You should have CUDA installed in your computer and all the GPU drivers must be set.

~~~~
git clone 'https://github.com/mozilla/TTS.git'
cd TTS
git checkout Tacotron2-iter-260K-824c091
python3 -m pip install -q gdown lws librosa Unidecode==0.4.20 tensorboardX git+git://github.com/bootphon/phonemizer@master localimport
cd ..
sudo apt-get install -y espeak
git clone 'https://github.com/erogol/WaveRNN.git'
cd WaveRNN
git checkout 8a1c152 
python3 -m pip install -r requirements.txt
cd ..
python3 -m pip install IPython
mkdir tts_models
mkdir wavernn_models
gdown -O wavernn_models/checkpoint_433000.pth.tar https://drive.google.com/uc?id=12GRFk5mcTDXqAdO5mR81E-DpTk8v2YS9
gdown -O wavernn_models/config.json https://drive.google.com/uc?id=1kiAGjq83wM3POG736GoyWOOcqwXhBulv
gdown -O tts_models/checkpoint_261000.pth.tar https://drive.google.com/uc?id=1otOqpixEsHf7SbOZIcttv3O7pG0EadDx
gdown -O tts_models/config.json https://drive.google.com/uc?id=1IJaGo0BdMQjbnCcOL4fPOieOEWMOsXE-
cd TTS
sudo python3 setup.py develop
~~~~
