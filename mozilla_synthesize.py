#!/usr/bin/env python3

from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.synthesis import synthesis
from TTS.utils.text import text_to_sequence, cleaners
from TTS.utils.generic_utils import load_config, setup_model
from TTS.utils.audio import AudioProcessor
from TTS.utils import *
from TTS.layers import *
from TTS.models.tacotron import Tacotron
from collections import OrderedDict
from pygame import mixer
import librosa.display
import os, sys, io, time, librosa, re, torch
import numpy as np

TTS_PATH = os.path.join('/home/at-home/Documents/atHome/TTS')
WAVERNN_PATH = os.path.join('/home/at-home/Documents/atHome/WaveRNN')

sys.path.append(TTS_PATH)  # set this if TTS is not installed globally
sys.path.append(WAVERNN_PATH)  # set this if TTS is not installed globally

iscuda = torch.cuda.is_available()
print('torch.cuda.is_available()=' + str(iscuda))
torch.cuda.empty_cache()

runcounter = 0

# Enter here the paths 
ROOT_PATH = TTS_PATH
MODEL_PATH = os.path.join(
    '/home/laura/Documents/at-home-tts/tts_models/checkpoint_261000.pth.tar')
CONFIG_PATH = os.path.join(
    '/home/laura/Documents/at-home-tts/tts_models/config.json')
OUT_FOLDER = os.path.join(ROOT_PATH, 'AudioSamples/benchmark_samples/')
CONFIG = load_config(CONFIG_PATH)
VOCODER_MODEL_PATH = os.path.join(
    '/home/laura/Documents/at-home-tts/wavernn_models/checkpoint_433000.pth.tar')
VOCODER_CONFIG_PATH = os.path.join(
    '/home/laura/Documents/at-home-tts/wavernn_models/config.json')
VOCODER_CONFIG = load_config(VOCODER_CONFIG_PATH)
# End of paths input

use_cuda = True

# Set the vocoder
use_gl = False  # use GL if True
batched_wavernn = True  # use batched wavernn inference if True

# LOAD TTS MODEL
num_chars = len(phonemes) if CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, CONFIG)

# load the audio processor
ap = AudioProcessor(**CONFIG.audio)

# load model state
if use_cuda:
    cp = torch.load(MODEL_PATH)
else:
    cp = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()
print(cp['step'])

# LOAD WAVERNN
if use_gl == False:
    from WaveRNN.models.wavernn import Model
    bits = 10

    wavernn = Model(
        rnn_dims=512,
        fc_dims=512,
        mode="mold",
        pad=2,
        upsample_factors=VOCODER_CONFIG.upsample_factors,  # set this depending on dataset
        feat_dims=VOCODER_CONFIG.audio["num_mels"],
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        hop_length=ap.hop_length,
        sample_rate=ap.sample_rate,
    ).cuda()

    check = torch.load(VOCODER_MODEL_PATH)
    wavernn.load_state_dict(check['model'])
    if use_cuda:
        wavernn.cuda()
    wavernn.eval()
    print(check['step'])

def custom_text_fix(sentence):
    global illegal_chars
    global repetition
    newsentence = illegal_chars.sub(' ', sentence)
    newsentence = repetition.sub(' ', newsentence)
    return newsentence

illegal_chars = re.compile(r'[^\w\d\.\,\;\!\?\s]')
repetition = re.compile(r'\s{2,}')

model.eval()
model.decoder.max_decoder_steps = 2000
speaker_id = 0

def mozilla_synthesize(sentence, filename):
    sentence = custom_text_fix(sentence)
    alizgn, spec, stop_tokens, wav = tts(
        model, sentence, CONFIG, use_cuda, ap, speaker_id=0, use_gl=use_gl, figures=True, filename=filename)


def tts(model, text, CONFIG, use_cuda, ap, use_gl, speaker_id=None, figures=True, filename="example.wav"):
    global runcounter
    t_1 = time.time()
    # submatch = re.sub(r'\s+',' ',text)
    # filenamematch = re.search( r'([^\s]+\s?\d+)',  submatch) # if filenamematch:
    #     filename = filenamematch.group(0) + '_' + str(runcounter) + '.wav'
    # else:
    #     filename = 'tempout_' + str(runcounter) + '.wav'

    runcounter += 1

    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens = synthesis(
        model, text, CONFIG, use_cuda, ap, truncated=False)
    if CONFIG.model == "Tacotron" and not use_gl:
        mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T).T
    if not use_gl:
        waveform = wavernn.generate(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(
            0).cuda(), batched=batched_wavernn, target=11000, overlap=550)

    print(" >  Run-time: {}".format(time.time() - t_1))
    os.makedirs(OUT_FOLDER, exist_ok=True)

    out_path = os.path.join(OUT_FOLDER, filename)
    ap.save_wav(waveform, out_path)
    return alignment, mel_postnet_spec, stop_tokens, waveform
