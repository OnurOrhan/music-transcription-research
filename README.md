## Machine Learning Methods for Music Transcription

This is the repository for my research about Music Transcription. In my report, I first explain the concept of transcription and its aspects. After that, I analyze some of the current methods for transcribing music and consider what more could be done.

My Neural Network model implemented in Keras, can predict the pitch of monophonic music with 80% accuracy per time frame. And currently, I am working on extending my model to polyphonic pitch detection. I will also implement a Hidden Markov Model for predicting the most probable note sequence.

<ul style="list-style-type:circle">
<li><b>src:</b> Source directory</li>
	<ul style="list-style-type:circle">
	  <li><b>figures:</b> Image files</li>
	  <li><b>midi:</b> Midi files, corresponding mp3 files and meta data as text files</li>
	  <li><b>output_midi:</b> Midi files generated from the model predictions.</li>
	  <li><b>scripts:</b> Python code files</li>
	  <li><b>test:</b> Example and test files</li>
	  <li><b>weights:</b> Trained model parameters</li>
	  <li><b>freqs.txt:</b> Frequency values of notes [E1-E5]</li>
	</ul>
<li><b>Workspace.ipynb:</b> Project material as a Jupyter notebook</li>
<li><b>...Report.pdf:</b> CMPE 492 Final Project Report</li>
</ul>
