#!/usr/bin/env python3
# -*- coding: utf-8 -*-

AUDIO_DIR = "/home/pablo/Dokumente/Uni/Bachelorarbeit/SELD_DCASE2020/TAU-NIGENS/mic_dev/testkopien/"
# Externe Festplatte
FEATURE_DIR = "/media/pablo/Elements/DCASE/Extracted_Features/"

#########################
# WENN DEBUG: Zu analysierender Bereich der WAV-Datei.  Wertebereich = [0,600]
STARTFRAME = 42
ENDFRAME = 48 # ENDFRAME selber nicht mehr enthalten
#########################

# 100 ms Frames -> frames/sec = 24000    
NUM = 2400 #samples per frame


# ANPASSBARE PARAMETER
#########################
# Anzahl Stützstellen für Beamforming (AZI:ELE ~ 2:1?)
NPOINTS_AZI = 70
NPOINTS_ELE = 35

FREQBANDS = [500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]



