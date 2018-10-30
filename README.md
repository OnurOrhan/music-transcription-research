## Machine Learning Methods for Music Transcription

This is a repository for my research about Music Transcription. I will first explain the concept of transcription and its aspects. As I move on, I will try to analyze current methods for transcribing music and consider what more could be done.

<b>Report.ipynb:</b> Project report as a Jupyter notebook.

<b>neural.py:</b> Python code for training and testing a neural network for monophonic audio transcription (88 piano frequency bins A0-C8)

<b>midi2mono.py:</b> Converts all MIDI files in "source/piano_midi" to monophonic MIDI files with prefixes "mono-". These files are saved into "source/piano_mono_midi"

<b>mono2mp3.py:</b> Converts all MIDI files in "source/piano_mono_midi" to MP3 audio files and saves them in this directory

<b>hmm.py:</b> Plots note transition probabilities of MIDI files in "source/piano_mono_midi"

<b>neural_guitar.py:</b> Old code for guitar samples (84 frequency bins C1-B7)
