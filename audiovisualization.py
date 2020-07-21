#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from wavimport import WavSamples
from os import scandir, path
from parameter import AUDIO_TEST, CSV_DIR, SINGLE_FILE_TEST, NUM
import matplotlib.pyplot as plt
import numpy as np
from acoular import L_p

framerate = 24000
soundpressure = np.zeros(600)
inactiveframes = np.array([])
activeframes = np.array([])

very_quiet_sound = 0
loud_silence = 0


with scandir(SINGLE_FILE_TEST) as files:
    for index, wavfile in enumerate(files):
        # zugehörige .csv-Datei, um aktive Frames zu plotten
        _name = wavfile.name[:-4]
        csvfile = CSV_DIR+_name+'.csv'
        csv_data = np.loadtxt(open(csvfile, "rb"), dtype="int32", delimiter=",", usecols=(0))

        ts = WavSamples(name = wavfile.path).data[:,3]
        print(wavfile.name)
        for frameindex, _ in enumerate(soundpressure):
            soundpressure[frameindex] = np.sum(ts[frameindex*NUM:(frameindex+1)*NUM]**2)/NUM
        frames = np.linspace(0, len(ts)/NUM, num=int(len(ts)/NUM))
        
        spl = L_p(soundpressure)


        fig1 = plt.subplot(211)
        fig1.axhline(55)
        plt.plot(frames, spl)
        plt.setp(fig1.get_xticklabels()) #, visible=False)
        plt.title(wavfile.name)

        fig2 = plt.subplot(212)
        fig2.axhline(0.04)
        plt.plot(frames, spl/np.linalg.norm(spl))
        # plt.yscale('log')
        # AKTIVE Frames: Plot bekommt Hintergrundfarbe
        # INAKTIVE Frames: Speichern der Energie des Frames zur Berechnung des Thresholds
        for frame in np.arange(600):
            if frame in csv_data:
                fig1.axvspan(frame, frame+1, facecolor='g', alpha=0.5)
                fig2.axvspan(frame, frame+1, facecolor='g', alpha=0.5)
                activeframes = np.append(activeframes, soundpressure[frame])
                if soundpressure[frame] < 0.000034:
                    very_quiet_sound += 1
            else:
                inactiveframes = np.append(inactiveframes, soundpressure[frame])
                if soundpressure[frame] > 0.000034:
                    loud_silence += 1
        plt.show()

print("Quiet Frames (Threshold to high) but with active Source = ", very_quiet_sound)
print("Loud Frames (Threshold to low) without active Source = ", loud_silence)
threshold = np.mean(inactiveframes)
print(threshold)
maxval = np.max(inactiveframes)
print(maxval)

print(np.min(activeframes))
print(np.mean(activeframes))


# histo = plt.hist(inactiveframes, bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram of inactive Frames")
# plt.xlim(xmax=0.001, xmin=0.0)
# plt.ylim(ymax=5000)
# # plt.yscale('log')
# plt.show()



# Folgender Code untersucht die Audio-Dateien, bezüglich deren erster aktiver Quelle
# => Da ca 1/6 der Dateien bereits innerhalb der ersten 5 Frames mind. 1 aktive Quelle
#    haben, fällt Kalibrierung durch "stillen" Anfangsteil weg.
        
# counter = 0
# with scandir(CSV_DIR) as files:
#     for index, file in enumerate(files):
#         csv_data = np.loadtxt(open(CSV_DIR+'/'+file.name, "rb"), dtype="int32", delimiter=",", usecols=(0))
#         for frame in csv_data:
#             if frame <= 4:
#                 print(file.name + "!!!")
#                 counter = counter + 1
#                 break
# print(counter)