import glob, os
import pretty_midi
import numpy as np

out_midis, out_names = ([], [])

# ----------------------------------------------------- #
# Reducing a midi file to a maximum of 3-note polyphony
# ----------------------------------------------------- #

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
    i, i1, i2 = (0, -1, -1) # Current index, and previous index for each channel

    pm.instruments[0].notes.sort(key=lambda x: x.start, reverse=False)
    for note in pm.instruments[0].notes: # Append the notes to the piano instrument
        start = note.start # The onset of the note
        end = note.end # The offset of the note
        pitch = note.pitch # The midi pitch of the note
        
        if end <= on1 and end <= on2: # Skip this note
            continue

        if start >= of1: # put note into the 1st slot
            p1 = pitch
            on1 = start
            of1 = end
            i1 = i # Final channel 1 index
        elif start >= of2: # put note into the 2nd slot
            p2 = pitch
            on2 = start
            of2 = end
            i2 = i # Final channel 2 index
        #elif pitch < p2 and pitch < p1:
        #    continue # Skip note if it has lower pitch than all three
        else: # If the previous note is going to be cropped
            index = -1 # Overlapping note index
            on = -1 # Overlapping note onset
            p = -1 # Overlapping note pitch
            
            if p2 < p1: # If channel 2 has lower pitch
                on = on2
                p = p2
                index = i2

                if on == start:
                    if pitch <= p:
                        continue
                else:
                    i2 = i # Final channel 2 index
                p2 = pitch
                on2 = start
                of2 = end

            else: # If channel 1 has lower pitch
                on = on1
                p = p1
                index = i1

                if on == start:
                    if pitch <= p:
                        continue
                else:
                    i1 = i # Final channel 1 index
                p1 = pitch
                on1 = start
                of1 = end

            if on == start and pitch > p: # If two notes start at the same time
                piano.notes[index] = note # Replace them
                continue
            else:
                piano.notes[index].end = start # then crop the previous note

        piano.notes.append(note)
        i += 1
        
    for pb in pm.instruments[0].pitch_bends: # Append pitch bend events to the instrument
        piano.pitch_bends.append(pb)
    for cc in pm.instruments[0].control_changes: # Append control changes to the instrument
        piano.control_changes.append(cc)

    piano_midi.instruments.append(piano) # Append the piano instrument to the midi file
    out_midis.append(piano_midi)
    out_names.append("poly-%s.mid" % filename)

# ------------------------------------ #
# Recording note events in a text file
# ------------------------------------ #

os.chdir("../piano_poly2_midi") # Switch to the monophonic midi folder directory 
print("Saving midi files...")

for zz, midi in enumerate(out_midis):
    midi.write(out_names[zz]) # Save the new monophonic midi files
    
    times = [0] # First collect all onset and offset times
    for note in midi.instruments[0].notes: # Going through the notes
        times.append(note.start)
        times.append(note.end)

    times = sorted(set(times)) # Sort timestamps and remove duplicates
    length = len(times) # Number of timestamps
    notes = np.full([length-1, 2], -1, dtype=np.int) # Pitch values per time frame
    polyphony = np.zeros(length-1, dtype=np.int) # Number of notes per time frame

    # Recording all the notes in the corresponding intervals
    for note in midi.instruments[0].notes:
        i = times.index(note.start)
        j = times.index(note.end)
        for k in range(i, j):
            p = polyphony[k]
            if p < 2 and note.pitch not in notes[k]:
                notes[k, p] = note.pitch
                polyphony[k] += 1

    file = open("%s.txt" % out_names[zz].split('.mid')[0], "w") # Text file with the note information
    for n, m in enumerate(notes):
        file.write("%d %d %f %f\n" % (m[0], m[1], times[n], times[n+1]))
    file.close()
    
os.chdir("../../scripts")
print("Done processing midi files!")
