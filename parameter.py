#!/usr/bin/env python3
# -*- coding: utf-8 -*-

AUDIO_DIR = "audio_csv_files/mic_dev/"
CSV_DIR = "audio_csv_files/metadata_dev/"
# Externe Festplatte
FEATURE_DIR = "extracted_features/" # "/media/pablo/Elements/DCASE/Extracted_Features/"

#########################
# DEBUG-Modus: --> Programm nutzt die Angabe von STARTFRAME & ENDFRAME, um den
#                  Bereich zu analysieren.
#              --> Wenn =False -> ganze Datei(en)
#              --> Wertebereich = [0,600]    1 Frame entspricht 100 ms
DEBUG = False

STARTFRAME = 42
ENDFRAME = 48 # ENDFRAME selber nicht mehr enthalten
#########################


JUST_1SOURCE_FRAMES = True



# DETAILEDINFO_LVL: 0 => keine Infos, nur aktueller Track
#                   1 => + Framenummern
#                   2 => + .csv-Werte des aktuellen Frames
#                   3 => + Frame-Ergebnisse
DETAILEDINFO_LVL = 2

# PLOTBEAMMAPS: Plottet 3D- und 2D-Ansicht des Frames. ACHTUNG: deutlich langsamer
PLOTBEAMMAPS = False



# 100 ms Frames -> frames/sec = 24000    
NUM = 2400 #samples per frame


# Anzahl Stützstellen für Beamforming (AZI:ELE ~ 2:1?)
NPOINTS_AZI = 70
NPOINTS_ELE = 35

# 4000 entfernt
FREQBANDS = [500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150]



