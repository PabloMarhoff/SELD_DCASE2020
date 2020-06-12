#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import scandir
from parameter import AUDIO_DIR, FEATURE_DIR, JUST_1SOURCE_FRAMES
from acoular import config
from fbeam_prep import fbeampreparation, audio_csv_extraction
from fbeam import fbeamextraction
from tf_helpers import write_TFRecord
from csv_reader import rm_2sources_frames
import time

###########################################################################
### Vor Ausführen   DEBUG,                                              ###
###                 DETAILEDINFO_LVL,                                   ###
###                 PLOTBEAMMAPS,                                       ###
###                 AUDIO_ und FEATURE_DIR in "parameter.py" anpassen   ###
###########################################################################


config.global_caching = "none" # "individual" #

# rg = Gridobj.
# st = SteeringVector
mg, rg, st, firstframe, lastframe = fbeampreparation()


with scandir(AUDIO_DIR) as files:
    t1 = time.time()
    for file in files:
        print("##############################")
        print("#",file.name, "#")
        print("##############################")
        _name = file.name[:-4]
        ts, be, csvdata = audio_csv_extraction(file.path, _name, st, firstframe)
        if JUST_1SOURCE_FRAMES:
            csvdata = rm_2sources_frames(csvdata)
        feature_matrix_pipeline = fbeamextraction(mg, rg, ts, be, firstframe, lastframe, csvdata)
        write_TFRecord(FEATURE_DIR+_name+".tfrecords",feature_matrix_pipeline,1) # write file (logs every written sample)
    t2 = time.time()
    print(t2-t1)
