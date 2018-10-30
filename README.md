## Machine Learning Methods for Music Transcription

This is a repository for my research about Music Transcription. I will first explain the concept of transcription and its aspects. As I move on, I will try to analyze current methods for transcribing music and consider what more could be done.

<b><u>Report.ipynb:</u></b> Project report as a Jupyter notebook.

<b><u>neural.py:</u></b> Python code for training and testing a neural network for monophonic audio transcription (88 piano frequency bins A0-C8)

<b><u>midi2mono.py:</u></b> Converts all MIDI files in "source/piano_midi" to monophonic MIDI files with prefixes "mono-". These files are saved into "source/piano_mono_midi"

<b><u>mono2mp3.py:</u></b> Converts all MIDI files in "source/piano_mono_midi" to MP3 audio files and saves them in this directory

<b><u>hmm.py:</u></b> Plots note transition probabilities of MIDI files in "source/piano_mono_midi"

<b><u>neural_guitar.py:</u></b> Old code for guitar samples (84 frequency bins C1-B7)
