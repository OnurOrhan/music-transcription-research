import os
import librosa
import numpy as np
import math
import random
from keras.layers import *
from keras.models import Model
from numpy.lib.stride_tricks import as_strided  
import matplotlib.pyplot as plt
# Reference #9: https://www.kaggle.com/asparago/simple-pitch-detector  

LEN = 1024
SR = 22050
FMAX = 11025
fbins = librosa.fft_frequencies(sr=SR)
freqs = dict()
x, y = ([], [])
model = None
# ----------
STD = 25 # Standard deviation for the probability vector
FREF = 10  # Reference frequency = 10 Hz

notes = {'Elo':['E2','F2','F#2/Gb2','G2','G#2/Ab2','A2','A#2/Bb2','B2','C3','C#3/Db3','D3','D#3/Eb3','E3'],
'A':['A2','A#2/Bb2','B2','C3','C#3/Db3','D3','D#3/Eb3','E3','F3','F#3/Gb3','G3','G#3/Ab3','A3'],
'D':['D3','D#3/Eb3','E3','F3','F#3/Gb3','G3','G#3/Ab3','A3','A#3/Bb3','B3','C4','C#4/Db4','D4'],
'G':['G3','G#3/Ab3','A3','A#3/Bb3','B3','C4','C#4/Db4','D4','D#4/Eb4','E4','F4','F#4/Gb4','G4'],
'B':['B3','C4','C#4/Db4','D4','D#4/Eb4','E4','F4','F#4/Gb4','G4','G#4/Ab4','A4','A#4/Bb4','B4'],
'Ehi':['E4','F4','F#4/Gb4','G4','G#4/Ab4','A4','A#4/Bb4','B4','C5','C#5/Db5','D5','D#5/Eb5','E5']}
testn = ['A2', 'B3', 'D3', 'E4', 'E2', 'G3', 'A1', 'D2', 'E1', 'G2']
testf = ['Ac1.mp3', 'Ac2.mp3', 'Ac3.mp3', 'Ac4.mp3', 'Ac5.mp3', 'Ac6.mp3', 'Ba1.mp3', 'Ba2.mp3', 'Ba3.mp3', 'Ba4.mp3']

def sampleWav(filename, freq):
    global x, y
    data, sampleRate = librosa.load(filename)
    D = librosa.stft(data)
    Spec = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.min)
    
    #pieces =  Spec.shape[1]
    #if pieces < 3: return
    #rand = random.sample(range(0, pieces), pieces) # Pick random frames
    
    pV = probVector(freq)
    temp = Spec[:, 22]
    x.append(temp)
    y.append(pV)
    
    #for i in range(len(rand)):
    #   x.append(Spec[:,rand[i]])
    #    y.append(pV)
    return

def freq2Index(freq):
    return int(round(1024*freq/FMAX))
    
def index2Freq(index): # Converting the vector index into cents
    return fbins[index]
    
def probVector(freq): # One-Hot frequency probability vector
    c = np.zeros(1025)
    
    for i in range(1025):
        if i == 0:
            c[i] = 0
        else:
            dif = 1200*math.log(fbins[i]/freq)/math.log(2) # Difference in cents
            c[i] = math.exp(dif**2/-2/STD**2)
    return c

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

def initSinusoids():
    global x, y
    x, y = ([], [])
    N = 1800
    phi = np.random.uniform(0, 2*math.pi, N)
    freq = np.random.exponential(240, N) + 32.7
    freq[freq > FMAX] = FMAX
    
    for i in range(N):
        x.append(np.sin(np.linspace(phi[i], phi[i] + freq[i]*LEN*2*math.pi/SR, LEN)))
        y.append(probVector(freq[i]))

    x = np.array(x)
    y = np.array(y)

def modelClear():
    global model
    model = None

def modelInit():
    global model

    if model is None:
        # Sequential model de denenebilir
        a = Input(shape=(1025,), name='input', dtype='float32')
        #b = Reshape(target_shape=(1025,), name='input-reshape')(a)
        
        # Sigmoid yerine relu?
        b = Dense(1025, activation='sigmoid', name="dense")(a)
        b = Dropout(0.25, name="dropout")(b)
        b = Dense(1025, activation='sigmoid', name="classifier")(b)

        model = Model(inputs=a, outputs=b)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

def oldModelInit():
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

def modelSummary():
    model.summary()

def modelFit(e=16, b=16, vs=0.2):
    l = len(x)
    lim = l*4//5
    rand = random.sample(range(0, l), l)
    model.fit(x=x[rand[:lim]], y=y[rand[:lim]], epochs=e, batch_size=b, validation_data=(x[rand[lim:]], y[rand[lim:]]))

def modelLoadWeights(filename, sinusoids=True):
    global model
    if model is None:
        if sinusoids:
            modelInitSinusoids()
        else:
            modelInit()
    model.load_weights(filename)
    model.compile('adam', 'binary_crossentropy')

def modelSaveWeights(filename):
    global model
    model.save_weights(filename)

def modelPredict(pred):
    return model.predict(pred)

def testFile(filename, freq):
    data, sampleRate = librosa.load(filename)
    D = librosa.stft(data)
    Spec = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.min)
    frames = []
    frames.append(Spec.mean(axis=1)) # Get the mean of the spectrum as input
    p = model.predict(np.array(frames))
    v = probVector(freq)

    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.title('%dHz -> Index: %d' % (index2Freq(p.argmax()), p.argmax()))
    plt.plot(p[0], 'r')
    plt.subplot(1,2,2)
    plt.title('%dHz -> Index: %d' % (freq,  freq2Index(freq)))
    plt.plot(v, 'k')
    plt.show()

def test(N=4):
    rand = random.sample(range(0, len(x)), N) # Pick random data
    a, b = (x[rand], y[rand])

    a = np.array(a)
    b = np.array(b)
    p = model.predict(a)

    plt.figure(figsize=(12, N*6))
    for i in range(N):
        plt.subplot(N,2,2*i+1)
        plt.plot(p[i], 'r')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Likelihood')
        f = p[i].argmax()
        plt.title('Prediction: %dHz, Index %d' % (index2Freq(f), f))

        plt.subplot(N,2,2*i+2)
        plt.plot(b[i], 'c')
        plt.xlabel('Frequency Bin')
        f = b[i].argmax()
        plt.title('Truth: %dHz, Index %d' % (index2Freq(f), f))
    plt.show()   

def testSinusoids():
    N = 5
    phi = np.random.uniform(0, 2*math.pi, N)
    freq = np.random.exponential(240, N) + 32.7
    freq[freq > FMAX] = FMAX
    a, b = ([], [])

    for i in range(N):
        a.append(np.sin(np.linspace(phi[i], phi[i] + freq[i]*2048*2*math.pi/SR, 2048, endpoint=False)))
        b.append(probVector(freq[i]))

    a = np.array(a)
    p = model.predict(a)

    plt.figure(figsize=(12, 24))
    for i in range(N):
        plt.subplot(4,2,2*i+1)
        plt.plot(p[i])
        f = freq[i]
        plt.xlabel('Frequency Bin')
        plt.title('%dHz -> Index %d' % (int(round(f)), freq2Index(f)))
        
        plt.subplot(4,2,2*i+2)
        plt.plot(b[i])
        plt.xlabel('Frequency Bin')
        plt.title('True Probability Vector')
    plt.show()
