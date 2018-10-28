import os
import librosa
import numpy as np
import math
import random
from keras.layers import *
from keras.models import Model
from numpy.lib.stride_tricks import as_strided  
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
# Reference #9: https://www.kaggle.com/asparago/simple-pitch-detector  

model = None
trainx, trainy = ([], [])
validx, validy = ([], [])
testx, testy = ([], [])

BINS = 84
FREF = librosa.note_to_hz('C1')  # Reference frequency = 32.70 Hz -> C1
fbins = librosa.cqt_frequencies(BINS, fmin=FREF)
FMAX = fbins[BINS-1]
freqs = dict()

STD = 25 # Standard deviation for the probability vector

# ----------
# SR = 22050
# FMAX = 11025
# LEN = 1024
# fbins = librosa.fft_frequencies(sr=SR)
# ----------

notes = ['Elo', 'A', 'D', 'G', 'B', 'Ehi']
all_notes = {'Elo':['E2','F2','F#2/Gb2','G2','G#2/Ab2','A2','A#2/Bb2','B2','C3','C#3/Db3','D3','D#3/Eb3','E3'],
                    'A':['A2','A#2/Bb2','B2','C3','C#3/Db3','D3','D#3/Eb3','E3','F3','F#3/Gb3','G3','G#3/Ab3','A3'],
            'D':['D3','D#3/Eb3','E3','F3','F#3/Gb3','G3','G#3/Ab3','A3','A#3/Bb3','B3','C4','C#4/Db4','D4'],
            'G':['G3','G#3/Ab3','A3','A#3/Bb3','B3','C4','C#4/Db4','D4','D#4/Eb4','E4','F4','F#4/Gb4','G4'],
                    'B':['B3','C4','C#4/Db4','D4','D#4/Eb4','E4','F4','F#4/Gb4','G4','G#4/Ab4','A4','A#4/Bb4','B4'],
                    'Ehi':['E4','F4','F#4/Gb4','G4','G#4/Ab4','A4','A#4/Bb4','B4','C5','C#5/Db5','D5','D#5/Eb5','E5']}

valid_notes = ['A#2/Bb2', 'C3', 'D#3/Eb3', 'G#3/Ab3', 'C4', 'F#4/Gb4']
#test_notes = ['A2', 'B3', 'D3', 'E4', 'E2', 'G3', 'A1', 'D2', 'E1', 'G2']
#test_files = ['Ac1.mp3', 'Ac2.mp3', 'Ac3.mp3', 'Ac4.mp3', 'Ac5.mp3', 'Ac6.mp3', 'Ba1.mp3', 'Ba2.mp3', 'Ba3.mp3', 'Ba4.mp3']
test_notes = ['A2', 'B3', 'D3', 'E4', 'E2', 'G3', 'G2']
test_files = ['Ac1.mp3', 'Ac2.mp3', 'Ac3.mp3', 'Ac4.mp3', 'Ac5.mp3', 'Ac6.mp3', 'Ba4.mp3']

def sampleWav(filename, freq, type=0):
    global trainx, trainy, validx, validy, testx, testy
    fraction = 2
    if(filename.startswith('Ac') or filename.startswith('Ba')):
        fraction = 3

    data, sampleRate = librosa.load(filename)
    length = int(len(data)/fraction)
    D = np.abs(librosa.cqt(data[:length], sr=sampleRate))
    Spec = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.min).T
    frames = len(Spec)

    z = 3
    if type == 2:
        z = 10

    rand = random.sample(range(0, frames), int(frames/z)) # Pick random data
    pV = np.tile(probVector(freq), (len(rand),1))

    #print("%s, total: %d, data_section: %d, frames: %d" % (filename, len(data), length, frames))

    if type == 0:
        if len(trainx) == 0:
            trainx = np.array(Spec[rand,:])
            trainy = pV
        else:
            trainx = np.append(trainx, Spec[rand,:], axis=0)
            trainy = np.append(trainy, pV, axis=0)
        
    elif type == 1:
        if len(validx) == 0:
            validx = np.array(Spec[rand,:])
            validy = pV
        else:
            validx = np.append(validx, Spec[rand,:], axis=0)
            validy = np.append(validy, pV, axis=0)

    elif type == 2:
        if len(testx) == 0:
            testx = np.array(Spec[rand,:])
            testy = pV
        else:
            testx = np.append(testx, Spec[rand,:], axis=0)
            testy = np.append(testy, pV, axis=0)   
    return

def freq2Index(freq):
    if freq < FREF:
        return 0
    elif freq > FMAX:
        return BINS-1
    else:
        return round(math.log2(freq/FREF) * 12)
    
def index2Freq(index): # Converting the vector index into cents
    return fbins[index]
    
def probVector(freq): # One-Hot frequency probability vector
    c = np.zeros(BINS)

    for i in range(BINS):
        dif = 1200*math.log2(fbins[i]/freq) # Difference in cents
        c[i] = math.exp(dif**2/-2/STD**2)
    return c

def initData():
    global freqs
    with open("source/guitar/freqs.txt") as f:
        for line in f:
            (key, val) = line.split()
            freqs[key] = float(val)      

    os.chdir("source/guitar")

    for index, string in enumerate(notes): # Dataset 1
        print("[%d/%d] %s processing..." % (index+1, len(notes), string))
        for i in range(9):
            filename = "%s%d.wav" % (string, i)
            note = all_notes[string][i] # Picking the note
            freq = freqs[note] # Picking the frequency value

            found = False # true (validation set), false (training set)
            for j in valid_notes:
                if j == note:
                    found = True

            if found:
                if freq <= freqs[all_notes[string][4]]:
                    sampleWav(filename, freq, 1)
            else:
                sampleWav(filename, freq, 0)
    
    print("Processing test files, please wait...")
    for i, f in enumerate(test_files): # Dataset 2
        freq = freqs[test_notes[i]]
        sampleWav(f, freq, 2)
    os.chdir("../..")

"""def initSinusoids():
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
    y = np.array(y)"""

def modelClear():
    global model
    model = None

def modelInit():
    global model

    if model is None:
        # Sequential model de denenebilir
        a = Input(shape=(BINS,), name='input', dtype='float32')
        #b = Reshape(target_shape=(BINS,), name='input-reshape')(a)
        
        # Sigmoid yerine relu?
        b = Dense(BINS, activation="relu", name="dense")(a)
        b = Dropout(0.15, name="dropout")(b)
        b = Dense(BINS, activation="sigmoid", name="dense2")(b)
        b = Dropout(0.15, name="dropout2")(b)
        b = Dense(BINS, activation="softmax", name="classifier")(b)

        model = Model(inputs=a, outputs=b)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

def modelSummary():
    model.summary()

def modelFit(e=16, b=32, v=0):
    model.fit(x=trainx, y=trainy, epochs=e, batch_size=b, verbose=v, validation_data=(validx, validy))

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
    D = np.abs(librosa.cqt(data, sr=sampleRate))
    Spec = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.min)
    
    #rand = random.sample(range(0, frames), int(frames/z)) # Pick random data
    #v = np.tile(probVector(freq), (len(rand),1))

    librosa.display.specshow(Spec, y_axis='cqt_hz', x_axis='time', cmap='magma')
    plt.title("Spectrogram of %s" % filename)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    ipd.Audio("source/alla_turca.wav")

    p = model.predict(np.array(Spec.T))
    librosa.display.specshow(p.T, y_axis='cqt_hz', x_axis='time', cmap='magma')
    plt.title("Prediction for %s" % filename)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    """plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.title('%dHz -> Index: %d' % (index2Freq(p.argmax()), p.argmax()))
    plt.plot(p[0], 'r')
    plt.subplot(1,2,2)
    plt.title('%dHz -> Index: %d' % (freq,  freq2Index(freq)))
    plt.plot(v, 'k')
    plt.show()"""

def test(N=3):
    if N > len(testx):
        N = len(testx)-1
    rand = random.sample(range(0, len(testx)), N) # Pick random data
    
    plt.figure(figsize=(12, N*6))
    for i in range(N):
        a, b = (testx[rand[i]], testy[rand[i]])
        a = np.array([a])
        b = np.array([b])
        p = model.predict(a)

        plt.subplot(N,2,2*i+1)
        plt.plot(p[0], 'r')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Likelihood')
        f = p[0].argmax()
        plt.title('Prediction: %dHz, Index %d' % (index2Freq(f), f))

        plt.subplot(N,2,2*i+2)
        plt.plot(b[0], 'c')
        plt.xlabel('Frequency Bin')
        f = b[0].argmax()
        plt.title('Truth: %dHz, Index %d' % (index2Freq(f), f))
    plt.show()

"""def testSinusoids():
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
    plt.show()"""
