import glob, os
import pretty_midi

out_midis, out_names = ([], [])

os.chdir("../midi/piano_midi")
for file in glob.glob("*.mid"):
    pm = pretty_midi.PrettyMIDI(file)
    filename = file.split('.mid')[0]
    print("Working on %s..." % filename)
    
    piano_midi = pretty_midi.PrettyMIDI() # Create the new monophonic midi file
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program) # Create a piano instrument
    
    p1, on1, of1 = (-1, -1, -1) # Last note in the  1st channel
    p2, on2, of2 = (-1, -1, -1) # Last note in the  2nd channel
    p3, on3, of3 = (-1, -1, -1) # Last note in the  3rd channel

    pm.instruments[0].notes.sort(key=lambda x: x.start, reverse=False)
    for note in pm.instruments[0].notes: # Append the notes to the piano instrument
        start = note.start # The onset of the note
        end = note.end # The offset of the note
        pitch = note.pitch
        
        if end <= of1 and end <= of2 and end <= of3: # Skip this note
            continue

        if start >= of1: # Could go into the first slot
            p1 = pitch
            on1 = start
            of1 = end
            piano.notes.append(note)
        elif start >= of2:
            p2 = pitch
            on2 = start
            of2 = end
            piano.notes.append(note)
        elif start >= of3:
            p3 = pitch
            on3 = start
            of3 = end
            piano.notes.append(note)
        else:
            

        """if(note.start == onset): # If two notes start at the same time
            if(note.pitch > pitch):
                piano.notes[len(piano.notes)-1] = note # Select the note with the higher pitch
            else:
                continue
        else:
            if(len(piano.notes) > 0 and note.start < offset): # If the previous note remains,
                piano.notes[len(piano.notes)-1].end = note.start # then crop the previous note
            piano.notes.append(note)
        pitch = note.pitch        
        onset = note.start
        offset = note.end"""
        
    for pb in pm.instruments[0].pitch_bends: # Append pitch bend events to the instrument
        piano.pitch_bends.append(pb)
    for cc in pm.instruments[0].control_changes: # Append control changes to the instrument
        piano.control_changes.append(cc)
        
    piano_midi.instruments.append(piano) # Append the piano instrument to the midi file
    out_midis.append(piano_midi)
    out_names.append("poly-%s.mid" % filename)

os.chdir("../piano_mono_midi") # Switch to the monophonic midi folder directory   
for j, midi in enumerate(out_midis):
    midi.write(out_names[j]) # Save the new monophonic midi files
    
    file = open("%s.txt" % out_names[j].split('.mid')[0], "w") # Text file with the note information
    for note in midi.instruments[0].notes: # Going through the notes
        file.write("%s " % pretty_midi.note_number_to_hz(note.pitch)) # Note frequency
        file.write("%s " % note.start) # Note onset
        file.write("%s\n" % note.end) # Note offset
    file.close()
    
os.chdir("../../scripts")
print("Done monophonisizing!")
