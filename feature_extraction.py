#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import scandir, path, remove
from parameter import AUDIO_DIR, FEATURE_DIR, FEATURE_DIR_TESTING_SPLIT,\
    SINGLE_FILE_TEST, JUST_1SOURCE_FRAMES, TRAINING, PLOTFILES, AUDIO_TEST,\
    DEBUG, THRESHOLD_FILTER
from acoular import config, h5cache
from fbeam_prep import fbeampreparation, audio_csv_extraction
from fbeam import fbeamextraction
from tf_helpers import write_TFRecord
import time
from numpy import save, load

###########################################################################
### Vor Ausführen   DEBUG,                                              ###
###                 DETAILEDINFO_LVL,                                   ###
###                 PLOTBEAMMAPS,                                       ###
###                 AUDIO_ und FEATURE_DIR in "parameter.py" anpassen   ###
###########################################################################

h5cache.cache_dir = path.join(path.curdir,'cache') # '/media/pablo/Elements/DCASE/Cache' # 
config.global_caching = "individual" # "none" # 

# rg = Gridobj.
# st = SteeringVector
mg, rg, st, firstframe, lastframe = fbeampreparation()

# wenn TRAINING, werden Frame-Infos aus WAV- und CSV-Dateien geladen
# wenn not(TRAINING), werden Frame-Infos nur aus WAV-Dateien extrahiert
with scandir(SINGLE_FILE_TEST) as files:
    t1 = time.time()
    for wavfile in files:
        print("##############################")
        print("#",wavfile.name, "#")
        print("##############################")
        _name = wavfile.name[:-4]
        
        # incl. "Threshold-Filtern" & "entfernen von Frames mit 2 Quellen"
        ts, be, labeldata = audio_csv_extraction(wavfile.path, _name, st, firstframe)

        # Hier Array mit allen aktiven, GEFILTERTEN Frames für Ergebnisse des KNN vorspeichern
        if not(TRAINING):
            with open(PLOTFILES+_name+'_frames.npy', mode='wb') as npyfile:
                save(npyfile, labeldata[:,0])

        # fbeamplot enthält alle durchs Maximum vom Beamforming bestimmten Richtungen
        fbeamplot = []
        fbeamplot_2ndSrc = []
        # algoplot enthält alle durch händischen Algo bestimmten Richtungen
        algoplot = []
        algoplot_2ndSrc = []
        feature_matrix_pipeline = fbeamextraction(mg, rg, ts, be,
                                                  firstframe, lastframe,
                                                  labeldata, _name,
                                                  fbeamplot, fbeamplot_2ndSrc,
                                                  algoplot, algoplot_2ndSrc)

        if TRAINING:
            write_TFRecord(FEATURE_DIR+_name+".tfrecords",feature_matrix_pipeline,1) # write file (logs every written sample)
        if not(TRAINING):
            write_TFRecord(FEATURE_DIR_TESTING_SPLIT+_name+".tfrecords",feature_matrix_pipeline,1)
        
        if not(DEBUG) and not(TRAINING):
            save(PLOTFILES+_name+"_beam.npy", fbeamplot)
            save(PLOTFILES+_name+"_algo.npy", algoplot)
            if fbeamplot_2ndSrc != []:
                save(PLOTFILES+_name+"_beam2nd.npy", fbeamplot_2ndSrc)
                save(PLOTFILES+_name+"_algo2nd.npy", algoplot_2ndSrc)
        if path.exists(h5cache.cache_dir+"/"+_name+"_cache.h5"):
            remove(h5cache.cache_dir+"/"+_name+"_cache.h5")
    t2 = time.time()
    print(t2-t1)
