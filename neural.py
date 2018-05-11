import os
import librosa
import numpy as np
import math
import random
from keras.layers import *
from keras.models import Model
from numpy.lib.stride_tricks import as_strided  
import matplotlib
# Reference #9: https://www.kaggle.com/asparago/simple-pitch-detector  

STD = 25 # 25 Cents
FREF = 10  # Reference frequency = 10 Hz
freqs = dict()
CSTEP = 20 # 20 Cent steps per vector element
C0 = 2051.1487628680293 # freq2Cents(32.70)
C = [i*CSTEP + C0 for i in range(360)]
LEN = 1024

x, y = ([], [])
model = None

notes = {'Elo':['E2','F2','F#2/Gb2','G2','G#2/Ab2','A2','A#2/Bb2','B2','C3','C#3/Db3','D3','D#3/Eb3','E3'],
'A':['A2','A#2/Bb2','B2','C3','C#3/Db3','D3','D#3/Eb3','E3','F3','F#3/Gb3','G3','G#3/Ab3','A3'],
'D':['D3','D#3/Eb3','E3','F3','F#3/Gb3','G3','G#3/Ab3','A3','A#3/Bb3','B3','C4','C#4/Db4','D4'],
'G':['G3','G#3/Ab3','A3','A#3/Bb3','B3','C4','C#4/Db4','D4','D#4/Eb4','E4','F4','F#4/Gb4','G4'],
'B':['B3','C4','C#4/Db4','D4','D#4/Eb4','E4','F4','F#4/Gb4','G4','G#4/Ab4','A4','A#4/Bb4','B4'],
'Ehi':['E4','F4','F#4/Gb4','G4','G#4/Ab4','A4','A#4/Bb4','B4','C5','C#5/Db5','D5','D#5/Eb5','E5']}
testn = ['A2', 'B3', 'D3', 'E4', 'E2', 'G3', 'A1', 'D2', 'E1', 'G2']
testf = ['Ac1.mp3', 'Ac2.mp3', 'Ac3.mp3', 'Ac4.mp3', 'Ac5.mp3', 'Ac6.mp3', 'Ba1.mp3', 'Ba2.mp3', 'Ba3.mp3', 'Ba4.mp3']

def sampleWav(filename, freq): # Splitting sound files into 1024 frame samples
    global x, y
    data, srate = librosa.load(filename, sr=16000)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    pieces =  (len(data) - LEN - 2000) // 100 + 1
    if pieces < 1: return
    rand = random.sample(range(0, pieces), pieces//20) # Pick random frames
    excerpts = [data[i*100+2000 : i*100+2000+LEN] for i in rand]

    pV = probVector(freq)
    for i in range(len(excerpts)):
        x.append(excerpts[i])
        y.append(pV)
    return

def freq2Cents(freq): # Converting frequencies into cents
    return 1200*math.log(freq/FREF)/math.log(2)

def freq2Index(freq):
    return int(round((freq2Cents(freq)-C0)/20))
    
def index2Freq(index): # Converting the vector index into cents
    return FREF*2**((index*CSTEP + C0)/1200)
    
def probVector(freq): # Probability density vector for a ground truth frequency
    c = freq2Cents(freq)
    return [math.exp(-(C[i]-c)**2*(1/2/STD**2)) for i in range(360)]

def init():
    global freqs, x, y
    x, y = ([], [])
    os.chdir('C:/Users/Onur/Desktop/music-transcription-research')
    with open("source/guitar/freqs.txt") as f:
        for line in f:
            (key, val) = line.split()
            freqs[key] = float(val)      

    os.chdir("source/guitar")
    for string in notes: # Dataset 1
        for i in range(9):
            filename = "%s%d.wav" % (string, i)
            note = notes[string][i] # Picking the note
            freq = freqs[note] # Picking the frequency value
            sampleWav(filename, freq)
    for i, f in enumerate(testf): # Dataset 2
        freq = freqs[testn[i]]
        sampleWav(f, freq)
    os.chdir("../..")

    x = np.array(x)
    y = np.array(y)

def modelInit():
    global model

    if model is None:
        model_capacity = 32
        layers = [1, 2, 3, 4, 5, 6]
        filters = [n * model_capacity for n in [32, 4, 4, 4, 8, 16]]
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        a = Input(shape=(1024,), name='input', dtype='float32')
        b = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(a)

        for layer, filters, width, strides in zip(layers, filters, widths, strides):
            b = Conv2D(filters, (width, 1), strides=strides, padding='same',
                       activation='relu', name="conv%d" % layer)(b)
            b = BatchNormalization(name="conv%d-BN" % layer)(b)
            b = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid',
                          name="conv%d-maxpool" % layer)(b)
            b = Dropout(0.25, name="conv%d-dropout" % layer)(b)

        b = Permute((2, 1, 3), name="transpose")(b)
        b = Flatten(name="flatten")(b)
        b = Dense(360, activation='sigmoid', name="classifier")(b)
        model = Model(inputs=a, outputs=b)
        model.compile('adam', 'binary_crossentropy')

def modelLoadWeights(filename):
    global model
    if model is None:
        modelInit()
    model.load_weights(filename)
    model.compile('adam', 'binary_crossentropy')

def modelSaveWeights(filename):
    global model
    model.save_weights(filename)

def modelFit(e, b):
    global model
    n = len(y)
    rand = random.sample(range(0, n), n) # Pick random frames
    limit = n*4//5 # Training data until this limit
    model.fit(x=x[rand[:limit]], y=y[rand[:limit]], epochs=e, batch_size=b, validation_data=(x[rand[limit:]], y[rand[limit:]]))

def modelPredict(filename):
    data, srate = librosa.load(filename, sr=16000)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    l = len(data)
    pieces =  (l - LEN) // 100 + 1
    excerpts = [data[i*100 : i*100+1024] for i in range(pieces)]

    frames = []
    for i in range(pieces):
        frames.append(excerpts[i])
    frames = np.array(frames)
    return model.predict(frames)
