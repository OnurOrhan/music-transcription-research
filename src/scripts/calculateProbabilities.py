import numpy as np
import math
import pretty_midi
import os, glob
import matplotlib.pyplot as plt
import librosa
import librosa.display

os.chdir("../midi/piano_poly_midi")
#for file in glob.glob("poly-chopin-etude-op10-no4"):
file = "poly-chopin-etude-op10-no4.txt"

# Assign acoustic state values to each time frame
# Calculate initial probabilities and transition probabilities
# Replicate these values for different chord combinations
# Save the probability matrices into the "src/data" directory

os.chdir("../../scripts")
