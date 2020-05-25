#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import scandir
from fbeam import fbeamextraction
from parameter import AUDIO_DIR, FEATURE_DIR
from tables import * # Für .h5-Export
# from numpy import save,load,savez

# debug-Modus: Programm nutzt die Angabe von STARTFRAME & ENDFRAME (parameter.py),
#              um den Bereich zu analysieren. Wenn =False -> ganze Datei(en)

# detailedinfo_lvl: 0 => keine Infos, nur aktueller Track
#                   1 => + Framenummern
#                   2 => + gerundetes Ergebnis aus allen Frames
#                   3 => + Frame-Ergebnisse

with scandir(AUDIO_DIR) as files:
    for file in files:
        print("##############################")
        print("#",file.name, "#")
        print("##############################")
        feature_matrix = fbeamextraction(file.path, debug=False, detailedinfo_lvl=1, plotbeammaps=False)
        #   h5file = open_file('fold1_room2_mix002_ov1.h5', mode='w', title='Audio Data')
        #   h5file.create_array('/', 'audio_features', obj=DL_Matrix,shape=DL_Matrix.shape)
        #   h5file.close()