import glob, os
import librosa
import numpy as np
import math
import random
import re
import pretty_midi
from keras.layers import *
from keras.models import Model
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

hPi = np.zeros(BINS) # HMM Initial state (note) probabilities
hSteps = np.zeros(BINS*2-1) # Transition steps
hA = np.zeros([BINS, BINS]) # Transition matrix
hB = np.zeros([BINS, BINS]) # Emission matrix

fbins = librosa.cqt_frequencies(BINS, fmin=FREF)
FMAX = fbins[BINS-1]
STD = 25 # Standard deviation for the probability vector

def sampleWav(filename, type=0):
    global trainx, trainy, validx, validy, hPi, hSteps
    
    data, Sr = librosa.load("%s.mp3" % filename)
    with open("%s.txt" % filename) as f:
        for line in f:
            (i1, i2, i3) = line.split()
            (a, b, c) = (float(i1), float(i2), float(i3))
            y = data[round(Sr*b) : round(Sr*c)]

            try:
                D = np.abs(librosa.cqt(y, sr=Sr, fmin=FREF, n_bins=BINS))
            except:
                continue
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

def freq2Index(freq): # Convert frequency to its frequency bin index
    if freq < FREF:
        return 0
    elif freq > FMAX:
        return BINS-1
    else:
        return round(math.log2(freq/FREF) * 12)

def midinote2Index(mid): # Convert piano midi note to cqt frequency bin index
# MIDI interval 21-108 corresponds to: CQT indexes 0-87
    if mid < 22:
        return 0
    elif mid > 107:
        return 87
    else:
        return mid - 21

def index2Midinote(i):
    return i+21
    
def index2Freq(index): # Converting the vector index into cents
    return fbins[index]

#def getSpecFrame(frame, sr): # Getting spectrum frame index, given the time-domain frame index
#    return 
    
def oneHotVector(freq): # One-hot probability vector
    c = np.zeros(BINS)
    c[freq2Index(freq)] = 1
    return c

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

    print("\nTraining samples: %d" % len(trainx))
    print("Validation samples: %d\n" % len(validx))

    os.chdir("../..")
    if np.isnan(trainx).any():
        print("Trainx has a NaN value!")
    if np.isnan(trainy).any():
        print("Trainy has a NaN value!")
    if np.isnan(validx).any():
        print("Validx has a NaN value!")
    if np.isnan(validy).any():
        print("Validy has a NaN value!")

    initTransitions()
    initEmissions()
    normalize()

def saveData():
    print("Saving data...")
    np.save("data/trainx.npy", trainx)
    np.save("data/trainy.npy", trainy)
    np.save("data/validx.npy", validx)
    np.save("data/validy.npy", validy)
    np.save("data/hPi.npy", hPi)
    np.save("data/hSteps.npy", hSteps)
    np.save("data/hA.npy", hA)
    np.save("data/hB.npy", hB)
    print("Data saved!")

def loadTrainData():
    global trainx, trainy, validx, validy
    print("Loading data...")
    trainx = np.load("data/trainx.npy")
    trainy = np.load("data/trainy.npy")
    validx = np.load("data/validx.npy")
    validy = np.load("data/validy.npy")
    print("Data loaded!")

def loadAllData():
    global trainx, trainy, validx, validy, hPi, hSteps, hA, hB
    print("Loading data...")
    trainx = np.load("data/trainx.npy")
    trainy = np.load("data/trainy.npy")
    validx = np.load("data/validx.npy")
    validy = np.load("data/validy.npy")
    hPi = np.load("data/hPi.npy")
    hSteps = np.load("data/hSteps.npy")
    hA = np.load("data/hA.npy")
    hB = np.load("data/hB.npy")
    print("Data loaded!")

def initTransitions():
    global hPi, hSteps, hA

    hPi = np.zeros(BINS)
    hSteps = np.zeros(BINS*2-1)
    hA = np.zeros([BINS, BINS])

    os.chdir("source/piano_mono_midi")
    for file in glob.glob("*.mid"):
        filename = file.split(".mid")[0]
        j = -1

        with open("%s.txt" % filename) as f:
            for line in f:
                (i1, i2, i3) = line.split()
                (a, b, c) = (float(i1), float(i2), float(i3))
                
                k = freq2Index(a)
                hPi[k] += 1
                if(j > -1):
                    difr = k - j
                    hSteps[difr+BINS-1] += 1
                j = k
    os.chdir("../..")

    hSteps += 1.0e-9
    hSteps = hSteps / np.sum(hSteps)
    initHA()
    
    print("Transition Matrix Complete!\n")

def initHA():
    global hA
    for i in range(BINS):
        hA[i] = hSteps[BINS-1-i : 2*BINS-1-i]

def initEmissions():
    global hB

    hB = np.zeros((BINS, BINS))
    print("Working on the Emission Matrix...")

    print("Predicting the training set...")
    T = model.predict(trainx)
    for i, x in enumerate(T):
        k = trainy[i].argmax()
        hB[k] += x

    print("Predicting the validation set...")
    V = model.predict(validx)
    for i, y in enumerate(V):
        k = validy[i].argmax()
        hB[k] += y

    print("Emission Matrix Complete!\n")

def normalize(): # Normalize all the probability matrices
    global hPi, hA, hB

    hPi += 1.0e-9
    hPi = hPi / np.sum(hPi)

    hA += 1.0e-9
    hA = hA / np.sum(hA)

    hB += 1.0e-9
    sumB = np.sum(hB, axis=1)
    hB = hB/np.tile(sumB, (BINS,1)).T

    print("Normalized all matrices!\n")

def modelClear(): # Clear the network model
    global model
    model = None

def modelInit(dropouts=[0.15, 0.15]): # Initialize the network model
    global model
    if model is None:
        a = Input(shape=(BINS,), name='input', dtype='float32')
        
        b = Dense(BINS*3-2, activation="relu", name="dense")(a)
        b = Dropout(dropouts[0], name="dropout")(b)
        b = Dense(BINS*3-2, activation="sigmoid", name="dense2")(b)
        b = Dropout(dropouts[1], name="dropout2")(b)
        b = Dense(BINS, activation="softmax", name="classifier")(b)

        model = Model(inputs=a, outputs=b)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

def modelSummary(): # Summarize the network model
    model.summary()

def modelFit(e=32, b=32, v=1):
    global history
    history = model.fit(x=trainx, y=trainy, epochs=e, batch_size=b, verbose=v, validation_data=(validx, validy))

def plotModel():
    global model
    if model is not None:
        plot_model(model, to_file='model.png')

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

def modelSaveWeights(filename): # Save current model weights as a file
    global model
    model.save_weights(filename)

def modelPredict(pred): # Make predictions for the given input spectrogram
    return model.predict(pred)

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
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, backtrack=True)
    onsets = np.append(onsets, [p.shape[0]-1])
    length = len(onsets)-1

    notes, starts, ends = ([], [], [])
    p = np.multiply(np.tile(np.sum(pp, axis=1), (BINS, 1)).T, p)
    
    back = np.zeros((length, BINS)) # Backtracking matrix
    prob = np.zeros((length, BINS)) # State Probabilities

    for i in range(length): # Viterbi Algorithm
        a = onsets[i]
        b = onsets[i+1]
        guesses = np.sum(p[a:b,:], axis=0)
        c = guesses.argmax()

        if i == 0: # The first note of the piece
            temp = np.multiply(hPi, hB[:, c])
            temp = temp / np.sum(temp)
            prob[0] = temp
        else: # The following notes in the piece
            for j in range(BINS):
                transitions = np.multiply(prob[i-1], hA[:, j])*hB[j, c]
                transitions = transitions / np.sum(transitions)
                index = transitions.argmax()
                back[i, j] = index
                prob[i, j] = transitions[index]             

        starts.append(a*512/sampleRate)
        ends.append(b*512/sampleRate)

    tran = np.zeros(p.shape)
    current = prob[length-1].argmax()
    tracking = np.array([current])
    
    for i in range(length-1): # Backtrack after Viterbi
        index = length-1-i
        
        temp = back[index, int(current)]
        tracking = np.insert(tracking, 0, temp)
        current = temp

    for i in range(length): # The final decision
        a = onsets[i]
        b = onsets[i+1]
        c = tracking[i]

        for j in range(a, b):
            tran[j, c] = 1

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
    librosa.display.specshow(Spec, y_axis='cqt_hz', x_axis='time', cmap='magma')
    plt.title("Spectrogram of %s" % filename, fontweight='bold')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    pp = Spec.T

    p = model.predict(np.array(pp))
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    librosa.display.specshow(p.T, y_axis='cqt_hz', x_axis='time')
    plt.colorbar()
    plt.title("Predictions", fontweight='bold')

     # First split into note segments, by detecting onsets
    oenv = librosa.onset.onset_strength(y=data, sr=sampleRate)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, backtrack=True)
    onsets = np.append(onsets, [p.shape[0]-1])
    length = len(onsets)-1

    notes, starts, ends = ([], [], [])
    p = np.multiply(np.tile(np.sum(pp, axis=1), (BINS, 1)).T, p)
    
    back = np.zeros((length, BINS)) # Backtracking matrix
    prob = np.zeros((length, BINS)) # State Probabilities

    for i in range(length): # Viterbi Algorithm
        a = onsets[i]
        b = onsets[i+1]
        guesses = np.sum(p[a:b,:], axis=0)
        c = guesses.argmax()

        if i == 0: # The first note of the piece
            temp = np.multiply(hPi, hB[:, c])
            temp = temp / np.sum(temp)
            prob[0] = temp
        else: # The following notes in the piece
            for j in range(BINS):
                transitions = np.multiply(prob[i-1], hA[:, j])*hB[j, c]
                transitions = transitions / np.sum(transitions)
                index = transitions.argmax()
                back[i, j] = index
                prob[i, j] = transitions[index]   

        starts.append(a*512/sampleRate)
        ends.append(b*512/sampleRate)

    tran = np.zeros(p.shape)
    current = prob[length-1].argmax()
    tracking = np.array([current])
    
    for i in range(length-1): # Backtrack after Viterbi
        index = length-1-i
        
        temp = back[index, int(current)]
        tracking = np.insert(tracking, 0, temp)
        current = temp

    for i in range(length): # The final decision
        a = onsets[i]
        b = onsets[i+1]
        c = tracking[i]

        for j in range(a, b):
            tran[j, c] = 1

        notes.append(c+21)

    plt.subplot(1,2,2)
    librosa.display.specshow(tran.T, y_axis='cqt_hz', x_axis='time')
    plt.colorbar()
    plt.title("Transcription", fontweight='bold')
    plt.show()

    createMidi(notes, starts, ends, filename)

def createMidi(notes, onsets, offsets, filename):
    piano_midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program) # Create a piano instrument
    
    for n, s, e in zip(notes, onsets, offsets):
        note = pretty_midi.Note(velocity=100, pitch=n, start=s, end=e)
        piano.notes.append(note)
        
    piano_midi.instruments.append(piano) # Append the piano instrument to the midi file
    piano_midi.write("out_midis/%s.mid" % (re.findall(r"[\w\-\_]+[/\\]?", filename)[-2]))