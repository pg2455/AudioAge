import pygame, time

FILE = 'data/two.wav'

################## play audio
def sync_playback(filename):
    # takes in a file and plays it back
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    time.sleep(5)

# sync_playback(FILE)

import soundfile as sf
import sounddevice as sd

def async_playback(filename):
    data, fs = sf.read(filename)
    sd.play(data, fs)
    return data, fs

# data, fs = async_playback(FILE)
# print(data.shape, fs)
# start = time.time()
# sd.wait()
# print("time for audio:", time.time() - start)

############ speech to text (pocketsphinx - not good)
import speech_recognition as sr_audio
import os

def transcribe_audio_sphinx(filename):
    # transcribe the audio (note this is only done if a voice sample)
    r=sr_audio.Recognizer()
    with sr_audio.AudioFile(filename) as source:
        audio = r.record(source)
    text=r.recognize_sphinx(audio)
    return text

# print(transcribe_audio_sphinx(FILE))

#(google API - needs credentials)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/pgupta/Dropbox/NeuroLex-5fa9a4dbbfa1.json"
def transcribe_audio_google(filename):
    # transcribe the audio (note this is only done if a voice sample)
    r=sr_audio.Recognizer()
    with sr_audio.AudioFile(filename) as source:
        audio = r.record(source)
    text=r.recognize_google_cloud(audio)
    return text

print(transcribe_audio_google(FILE))

## Train

# featurize and train a cnn
input_length = 16000*5 # 10 sec audio
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
model.fc = torch.nn.Linear(512, 7, bias = True)


loss_fn = nn.BCEWithLogitsLoss()
from scipy import signal

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+0.0001)
    return data-0.5

dir = "../Common_voice/Common_Voice_train/male/ages/male_teens"
class_dict = {'teens':0, 'twenties':1, 'thirties':2, 'forties':3, 'fifties':4, 'sixties':5, 'seventies':7}
class_ = class_dict[dir.split("_")[-1]]

from base_utils import parse_soundfile
for _file in os.listdir(dir):
    filename = os.path.join(dir, _file)
    data, fs = sf.read(filename)

    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset: (input_length + offset)]
    else:
        max_offset = input_length - len(data)
        offset = np.random.randint(max_offset)
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    data = audio_norm(data)
    print(data.max(), data.min())
    # sd.play(data, fs)
    # sd.wait()

    f, t, Sxx = signal.spectrogram(data, window=signal.gaussian(50, std=1))
    print(f.shape, t.shape, Sxx.shape, data.shape, fs)
    Sxx = torch.Tensor(Sxx).unsqueeze(0).unsqueeze(0)
    out = model(Sxx)
    target = torch.zeros(1, 7)
    target[0,class_ ] = 1.0
    print(out)

    print(loss_fn(out, target))


    Sxx2 = parse_soundfile(filename, 20, signal.gaussian(100, std=1) )
    print(Sxx2.shape)


    # t,id = signal.istft(Sxx)
    # sd.play(id)
