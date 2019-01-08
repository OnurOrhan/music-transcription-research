import glob, os
import librosa
import numpy as np
import math
import random
import re
import pretty_midi
from keras.layers import *
from keras.models import Model
from keras import optimizers
from keras.utils import plot_model
from numpy.lib.stride_tricks import as_strided  
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
# Reference [d]: Classical MIDI Files, https://www.mfiles.co.uk/classical-midi.htm

model = None
history = None
trainx, trainy = ([], [])
validx, validy = ([], [])

BINS = 88
FREF = librosa.note_to_hz('A0')  # Reference frequency = 27.50 Hz -> A0
# For 84 frequency bins:
# FREF = librosa.note_to_hz('C1')  # Reference frequency = 32.70 Hz -> C1

fbins = librosa.cqt_frequencies(BINS, fmin=FREF)
FMAX = fbins[BINS-1]
STD = 25 # Standard deviation for the probability vector

def sample(filename):
    global trainx, trainy, validx, validy
    
    data, Sr = librosa.load("%s.mp3" % filename)
    D = np.abs(librosa.cqt(data, sr=Sr, fmin=FREF, n_bins=BINS))
    Spec = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.min).T
    
    num_samples = 0 # Number of time frame samples per sound file

    with open("%s.txt" % filename) as f:
        for line in f:
            (i1, i2, i3) = line.split()
            num_samples += abs(int(float(i3)*Sr/512) - int(float(i2)*Sr/512))

    samples = np.zeros((num_samples, 88))
    pV = np.zeros((num_samples, 88))
    sample = 0

    with open("%s.txt" % filename) as f:
        for line in f:
            (i1, i2, i3) = line.split()
            (a, b, c) = (midinote2Index(int(i1)), float(i2), float(i3))
            
            y = np.array(Spec[int(b*Sr/512):int(c*Sr/512), :])
            length = len(y)
            ymin = y.min(axis=1)
            ymax = y.max(axis=1)
            for m in range(length):
                y[m] = (y[m]-ymin[m]) / (ymax[m] - ymin[m] + 1e-6) # Normalize spectrogram rows
            
            samples[sample:sample+length] = y
            pV[sample:sample+length] = np.tile(probVector(a), (length, 1))
            sample += length

    if random.random() < 0.8: # Training set or validation set?
        if len(trainx) == 0:
            trainx = samples
            trainy = pV
        else:
            trainx = np.append(trainx, samples, axis=0)
            trainy = np.append(trainy, pV, axis=0)
    else:
        if len(validx) == 0:
            validx = samples
            validy = pV
        else:
            validx = np.append(validx, samples, axis=0)
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
# MIDI interval 21-108 corresponds to: CQT indexes 0-87
    if mid < 21:
        return -1
    elif mid > 107:
        return 87
    else:
        return mid - 21

def index2Midinote(i):
    return i + 21
    
def oneHotVector(index): # One-hot probability vector
    c = np.zeros(BINS)
    c[index] = 1
    return c

def probVector(index): # Gaussian probability vector
    c = np.zeros(BINS)

    for i in range(BINS):
        dif = abs(1200*math.log2(fbins[i]/fbins[index])) # Difference in cents
        c[i] = math.exp(dif**2/-2/STD**2)
    return c

def initData(): # Initialize the training and validation data
    os.chdir("../midi/piano_mono_midi")

    files = []
    for file in glob.glob("*.mid"):
        files.append(file.split('.mid')[0]) # Get the list of file names
    total = len(files)

    print("Preparing the training and validation sets:")
    for i in range(total):
        print("[%d/%d] %s..." % (i+1, total, files[i]))
        sample(files[i]) # Add samples for both training & validation

    print("\nTraining samples: %d" % len(trainx))
    print("Validation samples: %d\n" % len(validx))

    os.chdir("../../scripts")

    if np.isnan(trainx).any():
        print("Trainx has a NaN value!")
    if np.isnan(trainy).any():
        print("Trainy has a NaN value!")
    if np.isnan(validx).any():
        print("Validx has a NaN value!")
    if np.isnan(validy).any():
        print("Validy has a NaN value!")

def saveData():
    print("Saving data...")
    np.save("../data/trainx.npy", trainx)
    np.save("../data/trainy.npy", trainy)
    np.save("../data/validx.npy", validx)
    np.save("../data/validy.npy", validy)
    print("Data saved...")

def loadData():
    global trainx, trainy, validx, validy
    print("Loading data...")
    trainx = np.load("../data/trainx.npy")
    trainy = np.load("../data/trainy.npy")
    validx = np.load("../data/validx.npy")
    validy = np.load("../data/validy.npy")
    print("Data loaded...")

def modelClear(): # Clear the network model
    global model
    model = None

def modelInit(dropouts=[0.15, 0.15]): # Initialize the network model
    global model
    if model is None:
        a = Input(shape=(BINS,), name='input', dtype='float32')
        
        b = Dense(BINS*5, activation="relu", name="dense1")(a)
        b = Dropout(dropouts[0], name="dropout1")(b)
        b = Dense(BINS*3, activation="sigmoid", name="dense2")(b)
        b = Dropout(dropouts[1], name="dropout2")(b)
        b = Dense(BINS, activation="softmax", name="classifier")(b)

        model = Model(inputs=a, outputs=b)
        model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

def modelSummary(): # Summarize the network model
    model.summary()

def modelFit(e=16, b=64, v=1):
    global history
    history = model.fit(x=trainx, y=trainy, epochs=e, batch_size=b, verbose=v, validation_data=(validx, validy))

def plotModel():
    global model
    if model is not None:
        plot_model(model, to_file='../figures/model.png')

def plotHistory():
    global history
    plt.plot(history.history['acc']) # Plot accuracy values
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()

    plt.plot(history.history['loss']) # Plot loss values
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

def modelLoadWeights(filename): # Load model weights from file
    global model
    if model is None:
        modelInit()
    model.load_weights(filename)
    model.compile('adam', 'binary_crossentropy')
    #history = np.load("../data/history.npy")

def modelSaveWeights(filename): # Save current model weights as a file
    global model
    model.save_weights(filename)
    np.save("../data/history.npy", history)

def modelPredict(pred): # Make predictions for the given input spectrogram
    return model.predict(pred)

def testNoteAccuracy():
    a = len(trainx)
    b = len(validx)

    p = model.predict(trainx)
    count = 0
    for i, x in enumerate(p):
        if trainy[i].argmax() == x.argmax():
            count += 1
    print("Training set: %d/%d correct. (%%%f)" % (count, a, 100.*count/a))

    pp = model.predict(validx)
    count2 = 0
    for i, x in enumerate(pp):
        if validy[i].argmax() == x.argmax():
            count2 += 1
    print("Validation set: %d / %d correct. (%%%f)" % (count2, b, 100.*count2/b))

def testFile(filename): # Test the given file
    data, sampleRate = librosa.load(filename)
    D = np.abs(librosa.cqt(data, sr=sampleRate, fmin=FREF, n_bins=BINS))
    Spec = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.min)

    plt.figure(figsize=(12, 10))
    plt.suptitle(filename)
    plt.subplot(2,2,1) 
    librosa.display.specshow(Spec, y_axis='cqt_hz', x_axis='time', cmap='magma')
    plt.title("Spectrogram", fontweight='bold')
    plt.colorbar(format='%+2.0f dB')

    frames = int(len(data)/512)
    z = np.zeros([frames, 88])
    with open("%s.txt" % filename.split('.')[0]) as f:
        for line in f:
            (i1, i2, i3) = line.split()
            (k, l, m) = (float(i1), float(i2), float(i3))
            freqq = freq2Index(k)
            for j in range(int(sampleRate*l/512), int(sampleRate*m/512)):
                z[j, freqq] = 1.0

    plt.subplot(2,2,2) 
    librosa.display.specshow(z.T, y_axis='cqt_hz', x_axis='time')
    plt.title("Ground Truth", fontweight='bold')

    pp = Spec.T

    p = model.predict(np.array(pp))
    plt.subplot(2,2,3)
    librosa.display.specshow(p.T, y_axis='cqt_hz', x_axis='time', label='Probabilities')
    plt.colorbar()
    plt.title("Predictions", fontweight='bold')

    # First split into note segments, by detecting onsets
    oenv = librosa.onset.onset_strength(y=data, sr=sampleRate)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sampleRate)
    onsets = np.append(onsets, [p.shape[0]-1])

    notes, starts, ends = ([], [], [])
    pp = np.multiply(np.tile(np.sum(pp, axis=1), (BINS, 1)).T, p)
    tran = np.zeros(p.shape)
    for i in range(len(onsets)-1):
        a = onsets[i]
        b = onsets[i+1]
        guesses = np.sum(pp[a:b,:], axis=0)
        c = guesses.argmax()
        
        for j in range(a, b):
            tran[j, c] = 1

        starts.append(a*512/sampleRate)
        ends.append(b*512/sampleRate)
        notes.append(c+21)

    plt.subplot(2,2,4)
    librosa.display.specshow(tran.T, y_axis='cqt_hz', x_axis='time')
    plt.title("Transcription", fontweight='bold')
    plt.show()

    createMidi(notes, starts, ends, filename)

def testFileQuick(filename): # Test the given file
    data, sampleRate = librosa.load(filename)
    D = np.abs(librosa.cqt(data, sr=sampleRate, fmin=FREF, n_bins=BINS))
    Spec = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.min)
    pp = Spec.T
    p = model.predict(np.array(pp))

    plt.figure(figsize=(10, 4))

    ax1 = plt.subplot(1, 2, 1)
    librosa.display.specshow(Spec, y_axis='cqt_hz', x_axis='time', cmap='magma')
    plt.colorbar()
    plt.title("Power spectrogram of %s" % filename, fontweight='bold')
  
    plt.subplot(1, 2, 2)
    librosa.display.specshow(p.T, y_axis='cqt_hz', x_axis='time')
    plt.colorbar()
    plt.title("Predictions", fontweight='bold')
    plt.show()
