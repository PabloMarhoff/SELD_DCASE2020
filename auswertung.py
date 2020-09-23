#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:45:02 2020

@author: pablo
"""

from acoular import PowerSpectra, BeamformerEig, L_p
from wavimport import WavSamples
from os import scandir, path, remove
from parameter import AUDIO_DIR, SINGLE_FILE_TEST, JUST_1SOURCE_FRAMES, NUM,\
    NPOINTS_ELE, NPOINTS_AZI, FREQBANDS, PLOTFILES, CSV_DIR, AUDIO_TEST
from fbeam_prep import rm_2sources_frames
from numpy import loadtxt, load, save, zeros, array, where, mean,\
    arange, sum as npsum, pi, sqrt,\
    sin, cos, clip, arccos, asarray
import matplotlib.pyplot as plt


# VOR AUSFÜHREN:
#   1. feature_extraction (mit TRAINING = False) ausführen
#   2. modeltraining (mit TRAINING = False) ausführen
#   -> Predictions mit KNN                  '_KNN.npy'
#   -> Predictions ohne KNN                 '_beam.npy'
#   -> mittels Threshold gefilterte Frames  '_frames.npy'

def smallest_LocError(ideal, real):
    dist = zeros(len(ideal))
    skipIt = False
    try:
        for i, frame in enumerate(ideal[:,0]):
            if skipIt:
                skipIt = False
                continue
            # Wenn in Frame 2 Quellen vorhanden:
            # 1. beide Abstandskombis ausrechnen  i|i & i+1|i+1  bzw. i|i+1 & i+1|i
            # 2. kleineren Fehler wählen
            if frame == ideal[i+1,0]:
                # nicht vertauscht
                dist_a1 = sin(ideal[i,2]) * sin(real[i,2]) + cos(ideal[i,2]) * cos(real[i,2]) * cos(abs(ideal[i,1] - real[i,1]))
                dist_a1 = arccos(dist_a1) * 180 / pi
                dist_a2 = sin(ideal[i+1,2]) * sin(real[i+1,2]) + cos(ideal[i+1,2]) * cos(real[i+1,2]) * cos(abs(ideal[i+1,1] - real[i+1,1]))
                dist_a2 = arccos(dist_a2) * 180 / pi
                # vertauscht
                dist_b1 = sin(ideal[i,2]) * sin(real[i+1,2]) + cos(ideal[i,2]) * cos(real[i+1,2]) * cos(abs(ideal[i,1] - real[i+1,1]))
                dist_b1 = arccos(dist_b1) * 180 / pi
                dist_b2 = sin(ideal[i+1,2]) * sin(real[i,2]) + cos(ideal[i+1,2]) * cos(real[i,2]) * cos(abs(ideal[i+1,1] - real[i,1]))
                dist_b2 = arccos(dist_b2) * 180 / pi
                if dist_a1+dist_a2 < dist_b1+dist_b2:
                    dist[i] = dist_a1
                    dist[i+1] = dist_a2
                else:
                    dist[i] = dist_b1
                    dist[i+1] = dist_b2
                skipIt = True
            # Wenn nur 1 Quelle, einfach Abstand bestimmen
            else:
                dist[i] = sin(ideal[i,2]) * sin(real[i,2]) + cos(ideal[i,2]) * cos(real[i,2]) * cos(abs(ideal[i,1] - real[i,1]))
                dist[i] = arccos(dist[i]) * 180 / pi
    except IndexError:
        pass
    return dist


def count_twoSrcFrames(csvdata, filteredFrames):
    tSF = 0
    for index, frame in enumerate(csvdata[:-1,0]):
        # ist Frame in den per Filter als aktiv bestimmten Frames enthalten
        if frame in filteredFrames[:]:
            # Wenn ja checken, ob 2 aktive Quellen
            if frame == csvdata[index+1,0]:
                tSF += 1
    return tSF

knn_dist_list = []
beam_dist_list = []
algo_dist_list = []


with scandir(SINGLE_FILE_TEST) as wavfiles:
    for file in wavfiles:
        _name = file.name[:-4]
        print('#######')
        print(_name)
        csvfile = CSV_DIR+_name+".csv"
        
        # csvdata enthält [Framenummer, Azi, Ele]
        # Wenn _ov2.wav (2 parallele Quellen möglich) sind jene Frames auch 2x in csvdata
        csvdata = loadtxt(open(csvfile, "rb"), dtype="int32", delimiter=",", usecols=(0,3,4))
        
        # TODO: wenn False funktioniert es noch nicht
        if JUST_1SOURCE_FRAMES:
            csvdata_2SF = csvdata
            save(PLOTFILES+_name+'_ideal_2SF.npy', csvdata_2SF)
            csvdata = rm_2sources_frames(csvdata)

        save(PLOTFILES+_name+'_ideal.npy', csvdata)
        
        # mit Soundpressure gefilterte Frames
        # --> ACHTUNG: kann Frames enthalten, in denen laut csv-Daten keine
        #     Quelle aktiv ist (fälschlicherweise)
        # --> Jeder Frame 1x oder 2x (bei 2 Quellen)
        filteredFrames = load(PLOTFILES+_name+'_frames.npy')
        
        # NICHT MEHR NÖTIG!?!
        # # Anzahl an Frames in filteredFrames mit 2 Quellen
        # tSF = count_twoSrcFrames(csvdata, filteredFrames)
        # print(tSF)
        

        # Wenn in der 4. Spalte eine '1' steht, ist der Thresholdpegel in diesem
        # Frame überschritten, obwohl laut csv-Datei keine aktive Quelle
        # vorhanden ist!!
        # [Frame, Azi, Ele, bool(Fehler)]
        filterideal = zeros((len(filteredFrames), 4))
        doubles = 0
        for fr_index, x in enumerate(filteredFrames[:]):
            if (x != filteredFrames[fr_index-1]):
                try:
                    csv_index = where(csvdata[:,0]==x)[0][0]
                    filterideal[fr_index,:3] = csvdata[csv_index,:3]
                    filterideal[fr_index,3] = 0
                    # Ende des Arrays 'csvdata'
                    if csv_index == len(csvdata)-1:
                        break
                    # 2. Quelle aktiv
                    if csvdata[csv_index+1,0] == x:
                        filterideal[fr_index+1,:3] = csvdata[csv_index+1,:3]
                        filterideal[fr_index+1,3] = 0
                # Wenn Frame über Threshold aber nicht in csv
                except IndexError:
                    filterideal[fr_index,0] = filteredFrames[fr_index]
                    filterideal[fr_index,1:3] = None
                    filterideal[fr_index,3] = 1

        save(PLOTFILES+_name+'_filterideal.npy', filterideal)

        beam = load(PLOTFILES+_name+'_beam.npy')
        algo = load(PLOTFILES+_name+'_algo.npy')
        knn = load(PLOTFILES+_name+'_KNN.npy')
        try:
            beam_2nd = load(PLOTFILES+_name+'_beam2nd.npy')
            algo_2nd = load(PLOTFILES+_name+'_algo2nd.npy')
            twoSrc_frames_exist = True
        except FileNotFoundError:
            twoSrc_frames_exist = False
        ideal = load(PLOTFILES+_name+'_ideal.npy')
        if JUST_1SOURCE_FRAMES:
            ideal_2SF = load(PLOTFILES+_name+'_ideal_2SF.npy')
        filterideal = load(PLOTFILES+_name+'_filterideal.npy')
        
        fig, axs = plt.subplots(5, 2)
        #links oben
        axs[0,0].plot(beam[:,0],beam[:,1], '.',ms=2)
        axs[0,0].set_title('Nur globales Maximum (Azimuth)', fontsize=8)
        axs[1,0].plot(algo[:,0], algo[:,1], '.', ms=2)
        axs[1,0].set_title('Algorithmus (Azimuth)', fontsize=8)
        axs[2,0].plot(knn[:,0], knn[:,1], '.', ms=2)
        axs[2,0].set_title('Mit KNN (Azimuth)', fontsize=8)
        axs[3,0].plot(filterideal[:,0], filterideal[:,1], '.', ms=2)
        axs[3,0].set_title('Ideal, mit Threshold gefiltert (Azimuth)', fontsize=8)
        #links unten
        axs[4,0].plot(ideal[:,0], ideal[:,1], '.', ms=2)
        axs[4,0].set_title('Ideal (Azimuth)', fontsize=8)

        # rechts oben
        axs[0,1].plot(beam[:,0],beam[:,2], '.',ms=2)
        axs[0,1].set_title('Nur globales Maximum (Elevation)', fontsize=8)
        axs[1,1].plot(algo[:,0], algo[:,2], '.', ms=2)
        axs[1,1].set_title('Algorithmus (Elevation)', fontsize=8)
        axs[2,1].plot(knn[:,0], knn[:,2], '.', ms=2)
        axs[2,1].set_title('Mit KNN (Elevation)', fontsize=8)
        axs[3,1].plot(filterideal[:,0], filterideal[:,2], '.', ms=2)
        axs[3,1].set_title('Ideal, mit Threshold gefiltert (Elevation)', fontsize=8)
        #rechts unten
        axs[4,1].plot(ideal[:,0], ideal[:,2], '.', ms=2)
        axs[4,1].set_title('Ideal (Elevation)', fontsize=8)

        if twoSrc_frames_exist: # 2. Quelle mit roten Punkten in obere
            axs[0,0].plot(beam_2nd[:,0], beam_2nd[:,1], '.', ms=2, color='r')
            axs[0,1].plot(beam_2nd[:,0], beam_2nd[:,2], '.', ms=2, color='r')
            axs[1,0].plot(algo_2nd[:,0], algo_2nd[:,1], '.', ms=2, color='r')
            axs[1,1].plot(algo_2nd[:,0], algo_2nd[:,2], '.', ms=2, color='r')
        if JUST_1SOURCE_FRAMES:
            axs[4,0].plot(ideal_2SF[:,0], ideal_2SF[:,1], '.', ms=2, c='r')
            axs[4,1].plot(ideal_2SF[:,0], ideal_2SF[:,2], '.', ms=2, c='r')
        
        for index, filterfault in enumerate(filterideal[:,0]):
            if bool(filterideal[index,3]):
                axs[1,0].axvspan(filterfault, filterfault+1, color='r', alpha=0.5, ec=None)
                axs[1,1].axvspan(filterfault, filterfault+1, color='r', alpha=0.5, ec=None)
                axs[4,0].axvspan(filterfault, filterfault+1, color='r', alpha=0.5, ec=None)
                axs[4,1].axvspan(filterfault, filterfault+1, color='r', alpha=0.5, ec=None)
        
        for x in arange(5):
            axs[x,0].set_ylim(-190,190)
            axs[x,0].set_xlim(-20,620)
        for x in arange(5):
            axs[x,1].set_ylim(-100,100)
            axs[x,1].set_xlim(-20,620)
        plt.subplots_adjust(left=0.05, right=0.96, hspace=0.7)
        fig.suptitle(file.name)
        plt.show()

#%% LOCALIZATION ERROR (°)

        filterideal[:,1:3] = filterideal[:,1:3] * pi/180
        knn[:,1:3] = knn[:,1:3] * pi/180

        # knn_dist = sin(filterideal[:,2]) * sin(knn[:,2]) + cos(filterideal[:,2]) * cos(knn[:,2]) * cos(abs(filterideal[:,1] - knn[:,1]))
        # TODO: Zeilen vertauschen, falls Frames verdreht bzw. knn_dist dann kleiner
        # Hier würde Kopplung Klassifizierung <-> Lokalisierung genutzt werden
        # Jetzt einfach kleineren Abstand nehmen
        
        knn_dist = smallest_LocError(filterideal, knn)
        knn_dist = mean(knn_dist)
        
        if twoSrc_frames_exist:
            # beam_AziEle_rad = beam und beam_2nd zusammengefasst
            # bzw. algo_AziEle_rad = ...
            algo_rad = zeros((len(beam)+len(beam_2nd),3))
            beam_rad = zeros((len(beam)+len(beam_2nd),3))
            ind_b2 = 0
            for ind_b1, fr in enumerate(beam[:,0]):
                beam_rad[ind_b1 + ind_b2] = beam[ind_b1,:]
                algo_rad[ind_b1 + ind_b2] = algo[ind_b1,:]
                if ind_b2 == len(beam_2nd):
                    continue
                elif beam_2nd[ind_b2,0] == fr:
                    beam_rad[ind_b1 + ind_b2 +1] = beam_2nd[ind_b2,:]
                    algo_rad[ind_b1 + ind_b2 +1] = algo_2nd[ind_b2,:]
                    ind_b2 += 1
        else:
            beam_rad = beam[:,:]
            algo_rad = algo[:,:]
        
        beam_rad[:,1:3] = beam_rad[:,1:3] * pi/180
        algo_rad[:,1:3] = algo_rad[:,1:3] * pi/180
        
        beam_dist = smallest_LocError(filterideal, beam_rad)
        beam_dist = mean(beam_dist)

        algo_dist = smallest_LocError(filterideal, algo_rad)
        algo_dist = mean(algo_dist)
        
        print("  Localization Error")
        print("    Just glob Max: ", beam_dist)
        print("    Algorithm:     ", algo_dist)
        print("    With ANN:      ", knn_dist)
        
        beam_dist_list.append(beam_dist)
        algo_dist_list.append(algo_dist)
        knn_dist_list.append(knn_dist)
    
    print('##########################')
    print('')
    print("Overall Localization Error")
    print("    Just glob Max: ", asarray(beam_dist_list).sum()/len(beam_dist_list))
    print("    Algorithm:     ", asarray(algo_dist_list).sum()/len(algo_dist_list))
    print("    With ANN:      ", asarray(knn_dist_list).sum()/len(knn_dist_list))



