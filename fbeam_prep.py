#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import sin, cos, array, deg2rad, zeros, sort, mean, loadtxt
from acoular import MicGeom, SteeringVector, PowerSpectra, BeamformerEig, L_p, BeamformerBase
from sphericalGrids import SphericalGrid_Equiangular
from wavimport import WavSamples
from parameter import NPOINTS_AZI, NPOINTS_ELE, DEBUG, DETAILEDINFO_LVL,\
    STARTFRAME, ENDFRAME, NUM, CSV_DIR, TRAINING, THRESHOLD_FILTER,\
        JUST_1SOURCE_FRAMES

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

def threshold_calculator(spl):
    spl_temp = sort(spl)
    threshold = mean(spl_temp[0:20])+8
    return threshold

#   --> WavSamples --> PowerSpectra --> BeamformerEig
#   --> Extraktion der .csv-Datei
def audio_csv_extraction(filepath, trackname, st, firstframe):

    if DETAILEDINFO_LVL >= 3:
        print('[Azimuth, Elevation], max_value')

    ts = WavSamples(name=filepath,start=firstframe*NUM,stop=(firstframe+1)*NUM)
    ps = PowerSpectra(time_data=ts, block_size=512, window='Hanning')
    be = BeamformerEig(freq_data=ps, steer=st, r_diag=True, n=3)


    # be = BeamformerBase(freq_data=ps, steer=st, r_diag=True)

    # bb = BeamformerBase(freq_data=ps, steer=st)
    # bo = BeamformerOrth(beamformer=be, eva_list=[3])
    
# Extraktion von Framenummer (Spalte 0), Geräuschklasse (Spalte 1), Azi (Spalte 3) und Ele (Spalte 4) aus .csv-Datei
    csvpath = CSV_DIR+trackname+".csv"
    rawdata = loadtxt(open(csvpath, "rb"), dtype="int32", delimiter=",", usecols=(0,1,3,4))


    if TRAINING or JUST_1SOURCE_FRAMES:
        labeldata = rawdata = rm_2sources_frames(rawdata)


    if THRESHOLD_FILTER:
        soundpressure = zeros((600,1))
        
    ## Sound Pressure Level ##
        wavsamples = WavSamples(name = filepath).data[:,3]
        for frameindex, _ in enumerate(soundpressure[:,0]):
            soundpressure[frameindex,0] = sum(wavsamples[frameindex*NUM:(frameindex+1)*NUM]**2)/NUM
        spl = L_p(soundpressure[:,0])
    ##########################

        threshold = threshold_calculator(spl)

    ## Liste der per Threshold gefilterten Frames ##
        active_frames = []
        for index, frame in enumerate(spl):
            if frame > threshold:
                active_frames.append(index)
    ################################################
        # active_frames weiterhin eine Liste, aus der man Elemente
        # auch zwischendrin löschen kann
        # thrdata = array(active_frames).reshape((len(active_frames),1))

        thrdata = []
        for ind, frame in enumerate(rawdata[:,0]):
            if frame in active_frames:
                thrdata.append(rawdata[ind,:])
            
        labeldata = array(thrdata)

    if not(TRAINING or THRESHOLD_FILTER):
        labeldata = rawdata


    return ts, be, labeldata




def rm_2sources_frames(rawdata):
    labeldata = zeros(rawdata.shape, "int32")
    doubles = 0
    for i, frame in enumerate(rawdata[:,0]):
        if rawdata[i-1,0] == frame:
            continue
        elif i+1 == len(rawdata):
            labeldata[i-2*doubles,:] = rawdata[i,:]
        elif rawdata[i+1,0] == frame:
            doubles = doubles + 1
        else:
            labeldata[i-2*doubles,:] = rawdata[i,:]
    return labeldata[:(len(labeldata)-2*doubles),:]

