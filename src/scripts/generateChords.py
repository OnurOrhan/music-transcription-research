import glob, os
import pretty_midi

INITIAL = 16384/16000
INCREMENT = 8192/16000

os.chdir("../midi/piano_chords")
# Generate chords with two notes
piano_midi = pretty_midi.PrettyMIDI()
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program) # Create a piano instrument
t = INITIAL

velocities = [49, 75, 101, 127]
limits = [4, 10, 18, 28, 40, 54, 70, 87]
prev = 0
for diff in range(1, 88): # iterate over the semitone difference of the two notes
    j = diff+21 # The initial chord: (0, diff)
    for i in range(21, 109-diff): # iterate over the first note range
        for v in velocities:
            piano.notes.append(pretty_midi.Note(velocity=v, pitch=i, start=t, end=t+INCREMENT))
            piano.notes.append(pretty_midi.Note(velocity=v, pitch=j, start=t, end=t+INCREMENT))
            t += INCREMENT
        j += 1
    
    if diff in limits:
        name = "%d-%d-chords" % (prev, diff)

        piano_midi.instruments.append(piano) # Append the piano instrument to the midi file
        piano_midi.write("%s.mid" % name)

        file = open("%s.txt" % name, "w") # Text file with the chord information
        first = True
        for note in piano_midi.instruments[0].notes: # Going through the notes
            file.write("%s " % pretty_midi.note_number_to_hz(note.pitch)) # Note frequency
            if not first:
                file.write("%s " % note.start) # Note onset
                file.write("%s\n" % note.end) # Note offset
                first = True
            else:
                first = False
        file.close()

        if limits[-1] != diff:
            prev = diff + 1
            piano_midi = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(program=piano_program)
            t = INITIAL
