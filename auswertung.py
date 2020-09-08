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

        knn = load(PLOTFILES+_name+'_KNN.npy')
        beam = load(PLOTFILES+_name+'_beam.npy')
        try:
            beam_2nd = load(PLOTFILES+_name+'_beam2nd.npy')
            twoSrc_frames_exist = True
        except FileNotFoundError:
            twoSrc_frames_exist = False
        ideal = load(PLOTFILES+_name+'_ideal.npy')
        if JUST_1SOURCE_FRAMES:
            ideal_2SF = load(PLOTFILES+_name+'_ideal_2SF.npy')
        filterideal = load(PLOTFILES+_name+'_filterideal.npy')
        
        fig, axs = plt.subplots(4, 2)
        axs[0,0].plot(knn[:,0], knn[:,1], '.', ms=2)
        axs[0,0].set_title('mit KNN (Azimuth)', fontsize=8)
        axs[1,0].plot(beam[:,0], beam[:,1], '.', ms=2)
        axs[1,0].set_title('ohne KNN (Azimuth)', fontsize=8)
        axs[2,0].plot(ideal[:,0], ideal[:,1], '.', ms=2)
        axs[2,0].set_title('ideal (Azimuth)', fontsize=8)
        axs[3,0].plot(filterideal[:,0], filterideal[:,1], '.', ms=2)
        axs[3,0].set_title('ideal, mit Threshold gefiltert (Azimuth)', fontsize=8)
        axs[0,1].plot(knn[:,0], knn[:,2], '.', ms=2)
        axs[0,1].set_title('mit KNN (Elevation)', fontsize=8)
        axs[1,1].plot(beam[:,0], beam[:,2], '.', ms=2)
        axs[1,1].set_title('ohne KNN (Elevation)', fontsize=8)
        axs[2,1].plot(ideal[:,0], ideal[:,2], '.', ms=2)
        axs[2,1].set_title('ideal (Elevation)', fontsize=8)
        axs[3,1].plot(filterideal[:,0], filterideal[:,2], '.', ms=2)
        axs[3,1].set_title('ideal, mit Threshold gefiltert (Elevation)', fontsize=8)
        if twoSrc_frames_exist:
            axs[1,0].plot(beam_2nd[:,0], beam_2nd[:,1], '.', ms=2, color='r')
            axs[1,1].plot(beam_2nd[:,0], beam_2nd[:,2], '.', ms=2, color='r')
        if JUST_1SOURCE_FRAMES:
            axs[2,0].plot(ideal_2SF[:,0], ideal_2SF[:,1], '.', ms=2, c='r')
            axs[2,1].plot(ideal_2SF[:,0], ideal_2SF[:,2], '.', ms=2, c='r')
        
        for index, filterfault in enumerate(filterideal[:,0]):
            if bool(filterideal[index,3]):
                axs[1,0].axvspan(filterfault, filterfault+1, color='r', alpha=0.5, ec=None)
                axs[1,1].axvspan(filterfault, filterfault+1, color='r', alpha=0.5, ec=None)
                axs[3,0].axvspan(filterfault, filterfault+1, color='r', alpha=0.5, ec=None)
                axs[3,1].axvspan(filterfault, filterfault+1, color='r', alpha=0.5, ec=None)
        
        for x in arange(4):
            axs[x,0].set_ylim(-190,190)
            axs[x,0].set_xlim(-20,620)
        for x in arange(4):
            axs[x,1].set_ylim(-100,100)
            axs[x,1].set_xlim(-20,620)
        plt.subplots_adjust(left=0.05, right=0.96, hspace=0.5)
        fig.suptitle(file.name)
        plt.show()
        
#%% LOCALIZATION ERROR (°)
        
        ideal_AziEle_rad = filterideal[:,1:3] * pi/180
        knn_AziEle_rad = knn[:,1:3] * pi/180

        knn_dist = sin(ideal_AziEle_rad[:,1]) * sin(knn_AziEle_rad[:,1]) + cos(ideal_AziEle_rad[:,1]) * cos(knn_AziEle_rad[:,1]) * cos(abs(ideal_AziEle_rad[:,0] - knn_AziEle_rad[:,0]))
        # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
        knn_dist = clip(knn_dist, -1, 1)
        knn_dist = arccos(knn_dist) * 180 / pi
        knn_dist = mean(knn_dist)
        
        if twoSrc_frames_exist:
            # beam_AziEle_rad = beam und beam_2nd zusammengefasst
            beam_AziEle_rad = zeros((len(beam)+len(beam_2nd),2))
            ind_b2 = 0
            for ind_b1, fr in enumerate(beam[:,0]):
                beam_AziEle_rad[ind_b1 + ind_b2] = beam[ind_b1,1:3] * pi/180
                if ind_b2 == len(beam_2nd):
                    continue
                elif beam_2nd[ind_b2,0] == fr:
                    beam_AziEle_rad[ind_b1 + ind_b2 +1] = beam_2nd[ind_b2,1:3] * pi/180
                    ind_b2 += 1
        else:
            beam_AziEle_rad = beam[:,1:3] * pi/180
        beam_dist = sin(ideal_AziEle_rad[:,1]) * sin(beam_AziEle_rad[:,1]) + cos(ideal_AziEle_rad[:,1]) * cos(beam_AziEle_rad[:,1]) * cos(abs(ideal_AziEle_rad[:,0] - beam_AziEle_rad[:,0]))
        # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
        beam_dist = clip(beam_dist, -1, 1)
        beam_dist = arccos(beam_dist) * 180 / pi
        beam_dist = mean(beam_dist)
        
        print("  Localization Error (with ANN): ", knn_dist)
        print("  Localization Error (no ANN): ", beam_dist)
        
        knn_dist_list.append(knn_dist)
        beam_dist_list.append(beam_dist)
    
    print('##########################')
    print('')
    print("Overall LocErr (with ANN): ", asarray(knn_dist_list).sum()/len(knn_dist_list))
    print("Overall LocErr (no ANN): ", asarray(beam_dist_list).sum()/len(beam_dist_list))


        # dif_knn_Azi = (knn_AziEle_rad[:,0] - ideal_AziEle_rad[:,0]) * 180/pi
        # dif_knn_Ele = (knn_AziEle_rad[:,1] - ideal_AziEle_rad[:,1]) * 180/pi
        # azi_error_knn = mean(sqrt(dif_knn_Azi[:] ** 2))
        # ele_error_knn = mean(sqrt(dif_knn_Ele[:] ** 2))
        # loc_error_knn = mean(sqrt(dif_knn_Azi[:] ** 2 + dif_knn_Ele[:] ** 2))

        # dif_beam_Azi = (beam_AziEle_rad[:,0] - ideal_AziEle_rad[:,0]) * 180/pi
        # dif_beam_Ele = (beam_AziEle_rad[:,1] - ideal_AziEle_rad[:,1]) * 180/pi
        # azi_error_beam = mean(sqrt(dif_beam_Azi[:] ** 2))
        # ele_error_beam = mean(sqrt(dif_beam_Ele[:] ** 2))
        # loc_error_beam = mean(sqrt(dif_beam_Azi[:] ** 2 + dif_beam_Ele[:] ** 2))

        