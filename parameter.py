#!/usr/bin/env python3
# -*- coding: utf-8 -*-

AUDIO_DIR = "audio_csv_files/mic_dev/"
AUDIO_DIR1 = "audio_csv_files/mic_dev_1/"
AUDIO_DIR2 = "audio_csv_files/mic_dev_2/"
AUDIO_DIR3 = "audio_csv_files/mic_dev_3/"
AUDIO_DIR4 = "audio_csv_files/mic_dev_4/"
CSV_DIR = "audio_csv_files/metadata_dev/"
PLOTFILES = "plotfiles/"

AUDIO_TEST = "audio_csv_files/audio_test/"
SINGLE_FILE_TEST = "audio_csv_files/single_file_test"

# Externe Festplatte
FEATURE_DIR = "extracted_features/" # "/media/pablo/Elements/DCASE/Extracted_Features/"

#########################
# Optionen für KNN
FEATURE_DIR_TRAINING_SPLIT = "extracted_features/training/"
FEATURE_DIR_TESTING_SPLIT = "extracted_features/testing/"
TRAINING = False
#########################


#########################
# DEBUG-Modus: --> Programm nutzt die Angabe von STARTFRAME & ENDFRAME, um den
#                  Bereich zu analysieren.
#              --> Wenn =False -> ganze Datei(en)
#              --> Wertebereich = [0,600]    1 Frame entspricht 100 ms
DEBUG = True

STARTFRAME = 275
ENDFRAME = 276 # ENDFRAME selber nicht mehr enthalten
#########################


JUST_1SOURCE_FRAMES = False
THRESHOLD_FILTER = True



# DETAILEDINFO_LVL: 0 => keine Infos, nur aktueller Track
#                   1 => + Framenummern
#                   2 => + .csv-Werte des aktuellen Frames
#                   3 => + Frame-Ergebnisse
DETAILEDINFO_LVL = 3

# PLOTBEAMMAPS: Plottet 3D- und 2D-Ansicht des Frames. ACHTUNG: deutlich langsamer
PLOTBEAMMAPS = True



# 100 ms Frames -> frames/sec = 24000    
NUM = 2400 #samples per frame


# Anzahl Stützstellen für Beamforming (AZI:ELE ~ 2:1?)
NPOINTS_AZI = 70
NPOINTS_ELE = 35

# 4000 entfernt
FREQBANDS = [500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150]



