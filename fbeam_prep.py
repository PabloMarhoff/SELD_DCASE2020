#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pylab import array, deg2rad
from numpy import sin, cos
from acoular import MicGeom, SteeringVector, PowerSpectra, BeamformerEig
from sphericalGrids import SphericalGrid_Equiangular
from wavimport import WavSamples
from csv_reader import csv_extractor
from parameter import NPOINTS_AZI, NPOINTS_ELE, DEBUG, DETAILEDINFO_LVL,\
    STARTFRAME, ENDFRAME, NUM, CSV_DIR

#%% EIN MAL zu Beginn des Programms
def fbeampreparation():
    # Gradangaben von Theta im Intervall [0,180] statt wie bei DCASE [90,-90]
    M1 = spherical2cart(deg2rad(45),deg2rad(55),0.042)
    M2 = spherical2cart(deg2rad(315),deg2rad(125),0.042)
    M3 = spherical2cart(deg2rad(135),deg2rad(125),0.042)
    M4 = spherical2cart(deg2rad(225),deg2rad(55),0.042)
    mg = MicGeom()
    mg.mpos_tot = array([M1,M2,M3,M4]).T # add microphone positions to MicGeom object
    
    # define evaluation grid
    rg = SphericalGrid_Equiangular(NPOINTS_AZI, NPOINTS_ELE)
    st = SteeringVector(grid=rg, mics=mg)
    
    if DEBUG:
        firstframe = STARTFRAME
        lastframe = ENDFRAME
    else:
        firstframe = 0
        lastframe = 600
    
    return mg, rg, st, firstframe, lastframe

# phi : azimuth
# theta : elevation
def spherical2cart(phi,theta,r):
    return array([
         r * sin(theta) * cos(phi),
         r * sin(theta) * sin(phi),
         r * cos(theta)
    ])


#%% Vor JEDER neuen Audio-Datei
#   --> WavSamples --> PowerSpectra --> BeamformerEig
#   --> Extraktion der .csv-Datei
def audio_csv_extraction(filepath, trackname, st, firstframe):

    if DETAILEDINFO_LVL >= 3:
        print('[Azimuth, Elevation], max_value')

    ts = WavSamples(name=filepath,start=firstframe*NUM,stop=(firstframe+1)*NUM)
    ps = PowerSpectra(time_data=ts, block_size=512, window='Hanning')
    be = BeamformerEig(freq_data=ps, steer=st, r_diag=True, n=3)
    # bb = BeamformerBase(freq_data=ps, steer=st)
    # bo = BeamformerOrth(beamformer=be, eva_list=[3])
    
    csvdata = csv_extractor(CSV_DIR+trackname+".csv")
    
    return ts, be, csvdata



