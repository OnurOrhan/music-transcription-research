import glob, os
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
# Reference #12: Classical MIDI Files, https://www.mfiles.co.uk/classical-midi.htm

model = None
trainx, trainy = ([], [])
validx, validy = ([], [])

BINS = 88
FREF = librosa.note_to_hz('A0')  # Reference frequency = 27.50 Hz -> A0
# For 84 frequency bins:
# FREF = librosa.note_to_hz('C1')  # Reference frequency = 32.70 Hz -> C1

fbins = librosa.cqt_frequencies(BINS, fmin=FREF)
FMAX = fbins[BINS-1]
STD = 25 # Standard deviation for the probability vector

def sampleWav(filename, type=0):
    global trainx, trainy, validx, validy
    freqs, onsets, offsets = ([], [], [])
    
    data, Sr = librosa.load("%s.mp3" % filename)
    
    with open("%s.txt" % filename) as f:
        for line in f:
            (i1, i2, i3) = line.split()
            (a, b, c) = (float(i1), float(i2), float(i3))
            y = data[round(Sr*b) : round(Sr*c)]
            D = np.abs(librosa.cqt(y, sr=Sr, fmin=FREF, n_bins=BINS))
            Spec = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.min).T
            pV = np.tile( probVector(a), (len(Spec),1) )

            if type == 0:
                if len(trainx) == 0:
                    trainx = np.array(Spec)
                    trainy = pV
                else:
                    trainx = np.append(trainx, Spec, axis=0)
                    trainy = np.append(trainy, pV, axis=0)
            elif type == 1:
                if len(validx) == 0:
                    validx = np.array(Spec)
                    validy = pV
                else:
                    validx = np.append(validx, Spec, axis=0)
                    validy = np.append(validy, pV, axis=0)
    return

def freq2Index(freq): # Convert frequency to its frequency bin index
    if freq < FREF:
        return 0
    elif freq > FMAX:
        return BINS-1
    else:
        return round(math.log2(freq/FREF) * 12)

def midinote2Index(mid): # Convert piano midi note to cqt frequency bin index
# MIDI: 21-108
# CQT: 0-87
    if mid < 22:
        return 0
    elif mid > 107:
        return 87
    else:
        return mid - 21
    
def index2Freq(index): # Converting the vector index into cents
    return fbins[index]
    
def probVector(freq): # Gaussian probability vector
    c = np.zeros(BINS)

    for i in range(BINS):
        dif = 1200*math.log2(fbins[i]/freq) # Difference in cents
        c[i] = math.exp(dif**2/-2/STD**2)
    return c

def initData(): # Initialize the training and validation data
    os.chdir("source/piano_mono_midi")

    files = []
    for file in glob.glob("*.mid"):
        files.append(file.split('.mid')[0]) # Get the list of file names
    total = len(files)
    n_train = round(total*0.8) # %80 / %20 train-validation split
    rand = random.sample(range(total), total)

    print("Preparing training set:")
    for i in range(n_train):
        print("[%d/%d] %s..." % (i+1, n_train, files[rand[i]]))
        sampleWav(files[rand[i]], 0) # Add samples for the training set

    print("Preparing validation set:")
    for j in range(n_train, total):
        print("[%d/%d] %s..." % (j-n_train+1, total-n_train, files[rand[j]]))
        sampleWav(files[rand[j]], 1) # Add samples for the validation set
    
    os.chdir("../..")

def modelClear(): # Clear the network model
    global model
    model = None

def modelInit(): # Initialize the network model
    global model

    if model is None:
        a = Input(shape=(BINS,), name='input', dtype='float32')
        
        b = Dense(BINS, activation="relu", name="dense")(a)
        b = Dropout(0.15, name="dropout")(b)
        b = Dense(BINS, activation="sigmoid", name="dense2")(b)
        b = Dropout(0.15, name="dropout2")(b)
        b = Dense(BINS, activation="softmax", name="classifier")(b)

        model = Model(inputs=a, outputs=b)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

def modelSummary(): # Summarize the network model
    model.summary()

def modelFit(e=100, b=32, v=1):
    model.fit(x=trainx, y=trainy, epochs=e, batch_size=b, verbose=v, validation_data=(validx, validy))

def modelLoadWeights(filename): # Load model weights from file
    global model
    if model is None:
        modelInit()
    model.load_weights(filename)
    model.compile('adam', 'binary_crossentropy')

def modelSaveWeights(filename): # Save current model weights as a file
    global model
    model.save_weights(filename)

def modelPredict(pred): # Make predictions for the given input spectrogram
    return model.predict(pred)

def testFile(filename): # Test the given file
    data, sampleRate = librosa.load(filename)
    D = np.abs(librosa.cqt(data, sr=sampleRate, fmin=FREF, n_bins=BINS))
    Spec = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.min)

    librosa.display.specshow(Spec, y_axis='cqt_hz', x_axis='time', cmap='magma')
    plt.title("Spectrogram of %s" % filename)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    ipd.Audio(filename)

    z = np.zeros([len(data), 88])
    with open("%s.txt" % filename.split('.')[0]) as f:
        for line in f:
            (i1, i2, i3) = line.split()
            (a, b, c) = (float(i1), float(i2), float(i3))
            freqq = freq2Index(a)
            for j in range(round(sampleRate*b), round(sampleRate*c)):
                z[j, freqq] = 1.0

    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1) 
    librosa.display.specshow(z.T, y_axis='cqt_hz', x_axis='time')
    plt.colorbar()
    plt.title("Ground truth")

    p = model.predict(np.array(Spec.T))
    plt.subplot(1,2,2) 
    librosa.display.specshow(p.T, y_axis='cqt_hz', x_axis='time')
    plt.colorbar()
    plt.title("Prediction for %s" % filename)
    plt.show()
