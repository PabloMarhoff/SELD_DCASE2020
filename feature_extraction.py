#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import scandir, path, remove
from parameter import AUDIO_DIR, FEATURE_DIR, SINGLE_FILE_TEST,\
    JUST_1SOURCE_FRAMES, TRAINING, PLOTFILES
from acoular import config, h5cache
from fbeam_prep import fbeampreparation, audio_csv_extraction
from fbeam import fbeamextraction
from tf_helpers import write_TFRecord
from csv_reader import rm_2sources_frames
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
        ts, be, csvdata = audio_csv_extraction(wavfile.path, _name, st, firstframe)

        if JUST_1SOURCE_FRAMES and TRAINING:
            csvdata = rm_2sources_frames(csvdata)

        # Hier Array mit allen aktiven, GEFILTERTEN Frames für Ergebnisse des KNN vorspeichern
        if not(TRAINING):
            with open(PLOTFILES+_name+'_frames.npy', mode='wb') as npyfile:
                save(npyfile, csvdata)
        # fbeamplot enthält alle durch Beamforming bestimmten Richtungen
        fbeamplot = []
        feature_matrix_pipeline = fbeamextraction(mg, rg, ts, be,
                                                  firstframe, lastframe,
                                                  csvdata, _name, fbeamplot)
        write_TFRecord(FEATURE_DIR+_name+".tfrecords",feature_matrix_pipeline,1) # write file (logs every written sample)
        
        save(PLOTFILES+_name+"_beam.npy", fbeamplot)
        if path.exists(h5cache.cache_dir+"/"+_name+"_cache.h5"):
            remove(h5cache.cache_dir+"/"+_name+"_cache.h5")
    t2 = time.time()
    print(t2-t1)
