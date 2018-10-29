import numpy as np
import math
import pretty_midi
import os, glob
import matplotlib.pyplot as plt

def demonstration():
    initial = []
    transitions = []
    transitions.append(dict())
    c = 0

    os.chdir("source/guitar_midi")
    for file in glob.glob("*.mid"):
        c += 1
        pm = pretty_midi.PrettyMIDI(file)
        transitions.append(dict())

        for instrument in pm.instruments:
            if not instrument.is_drum:
                initial.append(instrument.notes[0].pitch)

                for i in range(len(instrument.notes)-1):
                    jump = instrument.notes[i+1].pitch - instrument.notes[i].pitch

                    if jump in transitions[c]:
                        transitions[c][jump] += 1
                    else:
                        transitions[c][jump] = 1
    os.chdir("../..")

    # transitions[0] is the average probability dictionary
    # transitions[1-length] are the probability dictionaries for each .midi file
    s = 0
    length = len(transitions)-1
    a = []
    for t in range(length):
        b = sum(transitions[t+1].values()) # Number of note transitions per midi file
        a.append(b)
        s += b

    mean = 0
    for t in range(length):
        for k in transitions[t+1]:
            if k in transitions[0]: # Normalizing the average probabilities
                temp = transitions[t+1][k]/s
                transitions[0][k] += temp
                mean += temp*k
            else:
                transitions[0][k] = transitions[t+1][k]/s
            # Normalizing probabilities for each midi file
            transitions[t+1][k] = transitions[t+1][k]/a[t]

    var = 0
    for k in transitions[0]:
        var += (transitions[0][k]* ((mean - k)**2))
    x = np.linspace(-25, 25, 100)
    y = np.exp(-np.square(x-mean)/(2*var)) / (np.sqrt(2*np.pi*var))

    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)   
    plt.ylim(0,0.16)
    plt.bar(transitions[1].keys(), transitions[1].values(), alpha=0.6)
    plt.bar(transitions[2].keys(), transitions[2].values(), color='r', alpha=0.6)
    plt.bar(transitions[3].keys(), transitions[3].values(), color='m', alpha=0.6)
    plt.bar(transitions[4].keys(), transitions[4].values(), color='g', alpha=0.6)
    plt.bar(transitions[5].keys(), transitions[5].values(), color='c', alpha=0.6)
    plt.xlim(-25,25)
    plt.xlabel('Transition Step Size')
    plt.ylabel('Probability')
    plt.title('All Pitch Transition Probabilities')

    plt.subplot(1,2,2)
    #plt.plot(x, norm.pdf(x, mean, std))
    plt.plot(x,y, 'r_')
    plt.bar(transitions[0].keys(), transitions[0].values(), color='#A01010')
    plt.ylim(0,0.16)
    plt.xlim(-25,25)
    plt.xlabel('Transition Step Size')
    plt.ylabel('Probability')
    plt.title('Average Pitch Transition Probabilities')
    plt.show()
