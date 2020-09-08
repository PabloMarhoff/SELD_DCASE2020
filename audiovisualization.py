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

# ACHTUNG: threshold-Funktion für feature_extraction in fbeam_prep.py!!
def threshold_calculator(spl):
    spl_temp = np.sort(spl)
    threshold = np.mean(spl_temp[0:20])+8
    #threshold2 = 0 np.mean(spl_temp[0:3])+10
    return threshold#, threshold2



with scandir(SINGLE_FILE_TEST) as files:
    for index, wavfile in enumerate(files):
        # zugehörige .csv-Datei, um aktive Frames zu plotten
        _name = wavfile.name[:-4]
        csvfile = CSV_DIR+_name+'.csv'
        csv_frames = np.loadtxt(open(csvfile, "rb"), dtype="int32", delimiter=",", usecols=(0))

        ts = WavSamples(name = wavfile.path).data[:,3]
        print(wavfile.name)
        for frameindex, _ in enumerate(soundpressure):
            soundpressure[frameindex] = np.sum(ts[frameindex*NUM:(frameindex+1)*NUM]**2)/NUM
        frames = np.linspace(0, len(ts)/NUM, num=int(len(ts)/NUM))
        
        spl = L_p(soundpressure)
        
        fig1 = plt.subplot(211)
        
        threshold = threshold_calculator(spl)
        fig1.axhline(threshold)
        #fig1.axhline(threshold2, c='r')
        plt.plot(frames, spl)
        plt.setp(fig1.get_xticklabels()) #, visible=False)
        plt.title(wavfile.name + " 0:20 " + str(threshold))

        fig2 = plt.subplot(212)
        fig2.axhline(0.04)
        plt.plot(frames, spl/np.linalg.norm(spl))
        # plt.yscale('log')
        # AKTIVE Frames: Plot bekommt Hintergrundfarbe
        # INAKTIVE Frames: Speichern der Energie des Frames zur Berechnung des Thresholds
        for frame in np.arange(600):
            if frame in csv_frames:
                fig1.axvspan(frame, frame+1, facecolor='g', alpha=0.5)
                fig2.axvspan(frame, frame+1, facecolor='g', alpha=0.5)
                activeframes = np.append(activeframes, spl[frame])
            else:
                inactiveframes = np.append(inactiveframes, spl[frame])
        plt.show()

# threshold = np.mean(inactiveframes)
# print(threshold)
# maxval = np.max(inactiveframes)
# print(maxval)

# print(np.min(activeframes))
# print(np.mean(activeframes))






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