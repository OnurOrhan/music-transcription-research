import glob, os

os.chdir("../midi/piano_poly2_midi")
for file in glob.glob("*.mid"):
    filename = file.split('.mid')[0]
    print("Processing %s..." % filename)
    os.system("timidity %s -Ow -o %s.wav" % (file, filename))
    os.system("ffmpeg -i %s.wav -ab 128k %s.mp3" % (filename, filename))
    os.system("del /f %s.wav" % filename)
print("Done synthesizing audio!")
    
    # /////
    
    # 351-second-example
     
    # 27.00 sec (Timidity) MIDI --> MP3
     
    # or

    # 04.30 sec (Timidity) MIDI --> WAV
    # 15.00 sec (FFmpeg) WAV --> MP3
    # 19.30 sec (Total) MIDI --> MP3

    # /////
