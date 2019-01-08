import numpy as np
import math
import pretty_midi
import os, glob
import matplotlib.pyplot as plt
import librosa
import librosa.display

BINS = 88

init = numpy.zeros(BINS+1)
trans = numpy.zeros((BINS+1, BINS+1))
emis = numpy.zeros((BINS+1, BINS+1))

os.chdir("../midi/piano_mono_midi")
for file in glob.glob("*.txt"):
    with open("%s.txt" % filename) as f:
    for line in f:
        (i1, i2, i3) = line.split()
        num_samples += abs(int(float(i3)*Sr/512) - int(float(i2)*Sr/512))



        
# Assign acoustic state values to each time frame
# Calculate initial probabilities and transition probabilities
# Replicate these values for different chord combinations
# Save the probability matrices into the "src/data" directory

os.chdir("../../scripts")
