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
    NPOINTS_ELE, NPOINTS_AZI, FREQBANDS, PLOTFILES, CSV_DIR
from csv_reader import rm_2sources_frames
from numpy import loadtxt, load, save, zeros, array, where, arange, sum as npsum
import matplotlib.pyplot as plt


# VOR AUSFÜHREN:
#   1. feature_extraction (mit TRAINING = False) ausführen
#   2.1 .tfrecords-Datei in 'testing'-Ordner verschieben
#   2.2 modeltraining (mit TRAINING = False) ausführen
#   -> Predictions mit KNN                  '_KNN.npy'
#   -> Predictions ohne KNN                 '_beam.npy'
#   -> mittels Threshold gefilterte Frames  '_frames.npy'

with scandir(SINGLE_FILE_TEST) as wavfiles:
    for file in wavfiles:
        _name = file.name[:-4]
        csvfile = CSV_DIR+_name+".csv"
        
        # csvdata enthält [Framenummer, Azi, Ele]
        csvdata = loadtxt(open(csvfile, "rb"), dtype="int32", delimiter=",", usecols=(0,3,4))

        # TODO: wenn False funktioniert es noch nicht
        if JUST_1SOURCE_FRAMES:
            csvdata = rm_2sources_frames(csvdata)

        save(PLOTFILES+_name+'_ideal.npy', csvdata)
        
        # mit Soundpressure gefilterte Frames
        # ACHTUNG: kann Frames enthalten, in denen laut csv-Daten keine Quelle
        # aktiv ist (fälschlicherweise)
        frames = load(PLOTFILES+_name+'_frames.npy')
        

        # Wenn in der 4. Spalte eine '1' steht, ist der Thresholdpegel in diesem
        # Frame überschritten, obwohl laut csv-Datei keine aktive Quelle
        # vorhanden ist!!
        # [Frame, Azi, Ele, bool(Fehler)]
        filterideal = zeros((len(frames), 4))
        filterideal[:,0] = frames[:,0]
        for fr_index, x in enumerate(frames[:,0]):
            try:
                csv_index = where(csvdata[:,0]==x)[0][0]
                filterideal[fr_index,1:3] = csvdata[csv_index,1:3]
                filterideal[fr_index,3] = 0
            except IndexError:
                filterideal[fr_index,3] = 1
        
        save(PLOTFILES+_name+'_filterideal.npy', filterideal)

        knn = load(PLOTFILES+_name+'_KNN.npy')
        beam = load(PLOTFILES+_name+'_beam.npy')
        ideal = load(PLOTFILES+_name+'_ideal.npy')
        filterideal = load(PLOTFILES+_name+'_filterideal.npy')
        
        time = arange(600)
        fig, axs = plt.subplots(4, 2)
        axs[0,0].plot(knn[:,0], knn[:,1], '+')
        axs[0,0].set_title('mit KNN (Azimuth)')
        axs[1,0].plot(beam[:,0], beam[:,1], '+')
        axs[1,0].set_title('ohne KNN (Azimuth)')
        axs[2,0].plot(ideal[:,0], ideal[:,1], '+')
        axs[2,0].set_title('ideal (Azimuth)')
        axs[3,0].plot(filterideal[:,0], filterideal[:,1], '+')
        axs[3,0].set_title('ideal, mit Threshold gefiltert (Azimuth)')
        axs[0,1].plot(knn[:,0], knn[:,2], '+')
        axs[0,1].set_title('mit KNN (Elevation)')
        axs[1,1].plot(beam[:,0], beam[:,2], '+')
        axs[1,1].set_title('ohne KNN (Elevation)')
        axs[2,1].plot(ideal[:,0], ideal[:,2], '+')
        axs[2,1].set_title('ideal (Elevation)')
        axs[3,1].plot(filterideal[:,0], filterideal[:,2], '+')
        axs[3,1].set_title('ideal, mit Threshold gefiltert (Elevation)')
        
        for x in arange(4):
            axs[x,0].set_ylim(-190,190)
        for x in arange(4):
            axs[x,1].set_ylim(-100,100)
        plt.show()

        
        
        
        