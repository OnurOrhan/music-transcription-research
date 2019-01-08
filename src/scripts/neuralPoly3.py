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
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
# Reference [d]: Classical MIDI Files, https://www.mfiles.co.uk/classical-midi.htm

model = None
history = None
trainx, trainy = ([], [])
validx, validy = ([], [])

BINS = 88 # Number of frequency bins, except for the silent note
FREF = librosa.note_to_hz('A0')  # Reference frequency = 27.50 Hz (A0)

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
            (i1, i2, i3, i4, i5) = line.split()
            num_samples += abs(int(float(i5)*Sr/512) - int(float(i4)*Sr/512))

    samples = np.zeros((num_samples, 88))
    pV = np.zeros((num_samples, 88))
    sample = 0

    with open("%s.txt" % filename) as f:
        for line in f:
            (i1, i2, i3, i4, i5) = line.split()
            (a, b, c, d, e) = (midinote2Index(int(i1)), midinote2Index(int(i2)), midinote2Index(int(i3)), float(i4), float(i5))
            
            y = np.array(Spec[int(d*Sr/512):int(e*Sr/512), :])
            length = len(y)
            ymin = y.min(axis=1)
            ymax = y.max(axis=1)
            for m in range(length):
                y[m] = (y[m]-ymin[m]) / (ymax[m] - ymin[m] + 1e-6) # Normalize spectrogram rows
            
            samples[sample:sample+length] = y
            pV[sample:sample+length] = np.tile(probVector([a,b,c]), (length, 1))
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
    
def oneHotVector(indexes): # One-hot probability vector
    c = np.zeros(BINS)
    for i in indexes:
        if i > -1:
            c[i] = 1
    return c

def probVector(indexes): # Gaussian probability vector
    c = np.zeros(BINS)
    ind = []
    for j in indexes:
        if j > -1:
            ind.append(j)

    for i in range(BINS):
        value = 0
        for k in ind:
            dif = abs(1200*math.log2(fbins[i]/fbins[k])) # Difference in cents
            value += math.exp(dif**2/-2/STD**2)
        
        if value > 1:
            value = 1
        c[i] = value
    return c

def initData(): # Initialize the training and validation data
    os.chdir("../midi/piano_poly3_midi")

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
    np.save("../data/3tx.npy", trainx)
    np.save("../data/3ty.npy", trainy)
    np.save("../data/3vx.npy", validx)
    np.save("../data/3vy.npy", validy)
    print("Data saved...")

def loadData():
    global trainx, trainy, validx, validy
    print("Loading data...")
    trainx = np.load("../data/3tx.npy")
    trainy = np.load("../data/3ty.npy")
    validx = np.load("../data/3vx.npy")
    validy = np.load("../data/3vy.npy")
    print("Data loaded...")

def modelClear(): # Clear the network model
    global model
    model = None

def modelInit(dropouts=[0.1, 0.1]): # Initialize the network model
    global model
    if model is None:
        a = Input(shape=(BINS,), name='input', dtype='float32')
        
        b = Dense(BINS*7, activation="relu", name="dense1")(a)
        b = Dropout(dropouts[0], name="dropout1")(b)
        b = Dense(BINS*5, activation="relu", name="dense2")(b)
        b = Dropout(dropouts[1], name="dropout2")(b)
        b = Dense(BINS, activation="sigmoid", name="classifier")(b)

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
        plot_model(model, to_file='../figures/model3.png')

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
    history = np.load("../data/history3.npy")

def modelSaveWeights(filename): # Save current model weights as a file
    global model
    model.save_weights(filename)
    np.save("../data/history3.npy", history)

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
    print("Validation set: %d/%d correct. (%%%f)" % (count2, b, 100.*count2/b))

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

    # First split into note segments, by detecting onsets
    oenv = librosa.onset.onset_strength(y=data, sr=sampleRate)
    times = librosa.frames_to_time(np.arange(len(oenv)), sr=sampleRate)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sampleRate)
    onsets = np.append(onsets, [p.shape[0]-1])

    plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(1, 2, 1)
    librosa.display.specshow(Spec, y_axis='cqt_hz', x_axis='time', cmap='magma')
    plt.title("Power spectrogram of %s" % filename, fontweight='bold')
  
    plt.subplot(1, 2, 2)
    plt.plot(times, oenv, label='Onset Strength')
    plt.vlines(times[onsets], 0, oenv.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    plt.axis('tight')
    plt.legend(frameon=True, framealpha=0.75)
    plt.show()
    
    librosa.display.specshow(p.T, y_axis='cqt_hz', x_axis='time')
    plt.colorbar()
    plt.title("Predictions", fontweight='bold')
    plt.show()

    """notes, starts, ends = ([], [], [])
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

    plt.subplot(2, 2, 4)
    librosa.display.specshow(tran.T, y_axis='cqt_hz', x_axis='time')
    plt.colorbar()
    plt.title("Transcription", fontweight='bold')
    plt.show()

    createMidi(notes, starts, ends, filename)"""

def createMidi(notes, onsets, offsets, filename):
    piano_midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program) # Create a piano instrument
    
    for n, s, e in zip(notes, onsets, offsets):
        note = pretty_midi.Note(velocity=100, pitch=n, start=s, end=e)
        piano.notes.append(note)
        
    piano_midi.instruments.append(piano) # Append the piano instrument to the midi file
    piano_midi.write("../output_midi/tr-%s.mid" % (re.findall(r"[\w\-\_]+[/\\]?", filename)[-2]))
