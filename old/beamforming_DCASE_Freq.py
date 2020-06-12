#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tables import * # Für .h5-Export
from os import path
from acoular import __file__ as bpath, MicGeom, WNoiseGenerator, PointSource,\
 Mixer, WriteH5, TimeSamples, PowerSpectra, RectGrid, SteeringVector,\
 BeamformerBase, L_p,SineGenerator, BeamformerTime, TimeAverage, TimePower,\
 FiltFiltOctave, WriteWAV, BeamformerClean, BeamformerTimeSq, BeamformerEig, BeamformerOrth
from pylab import figure, plot, axis, imshow, colorbar, show,title, zeros,\
    deg2rad,rad2deg,array
from wavimport import WavSamples
from sphericalGrids import SphericalGrid_Equiangular, SphericalGrid_EvenlyDist
from numpy import argmax,amax,pi,arange,arctan2,sin,cos,arccos,sqrt,zeros,save,load,savez
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# 100 ms Frames -> frames/sec = 24000    
NUM = 2400 #samples per frame
AUDIO_DIR = "/home/pablo/Dokumente/Uni/Bachelorarbeit/SELD_DCASE2020/TAU-NIGENS/mic_dev/"

# TODO Funktion scheiben zum Iterieren durch alle Dateien
TRACK = "fold1_room2_mix002_ov1.wav"
# Externe Festplatte
FEATURE_DIR = "/media/pablo/Elements/DCASE/Extracted_Features/"

# Anzahl Stützstellen für Beamforming (AZI:ELE ~ 2:1?)
NPOINTS_AZI = 70
NPOINTS_ELE = 35

# Zu analysierender Bereich der WAV-Datei.  Wertebereich = [0,600]
STARTFRAME = 42
ENDFRAME = 48 # ENDFRAME selber nicht mehr enthalten
# Anzahl der zu analysierenden Frames
FRAMES = ENDFRAME - STARTFRAME

FREQBANDS = [500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]



def spherical2cart(phi,theta,r):
    return array([
         r * sin(theta) * cos(phi),
         r * sin(theta) * sin(phi),
         r * cos(theta)
    ])
# Gradangaben von Theta im Intervall [0,180] statt wie bei DCASE [90,-90]
M1 = spherical2cart(deg2rad(45),deg2rad(55),0.042)
M2 = spherical2cart(deg2rad(315),deg2rad(125),0.042)
M3 = spherical2cart(deg2rad(135),deg2rad(125),0.042)
M4 = spherical2cart(deg2rad(225),deg2rad(55),0.042)




mg = MicGeom()
mg.mpos_tot = array([M1,M2,M3,M4]).T # add microphone positions to MicGeom object

# define evaluation grid
rg = SphericalGrid_Equiangular(NPOINTS_AZI, NPOINTS_ELE)
   
# analyze the data and generate map
name = AUDIO_DIR+TRACK

x = rg.gpos[0]
y = rg.gpos[1]
z = rg.gpos[2]

summed_azi_x = 0.0
summed_azi_y = 0.0
summed_ele = 0.0
summed_max_v = 0.0
print('[Azimuth, Elevation], max_value')

st = SteeringVector(grid=rg, mics=mg)
ts = WavSamples(name=name,start=STARTFRAME*NUM,stop=(STARTFRAME+1)*NUM)
ps = PowerSpectra(time_data=ts, block_size=512, window='Hanning')
# bb = BeamformerBase(freq_data=ps, steer=st)

be = BeamformerEig(freq_data=ps, steer=st, r_diag=True, n=3)
# bo = BeamformerOrth(beamformer=be, eva_list=[3])

##################### DeepLearning-Matrix ##########################
################   Elevation    Azimuth      Frequenzbänder  Frames
DL_Matrix = zeros((NPOINTS_ELE, NPOINTS_AZI, len(FREQBANDS), FRAMES))
# Wenn .npz, dann keine FRAMES, weil die in einzelne .npy-Tabellen?!
# DL_Matrix = zeros((NPOINTS_ELE, NPOINTS_AZI, len(FREQBANDS)))


for frame_index, frame in enumerate(range(STARTFRAME, ENDFRAME)):
    print('### FRAME: ', frame-STARTFRAME, ' (',frame,') ###')
    for freq_index, freq in enumerate(FREQBANDS):
        print('FREQ =', freq)
        ts.start = frame*NUM
        ts.stop = (frame+1)*NUM
    
        result = zeros((4,rg.shape[0],rg.shape[1]))
        for i in range(4):
            be.n = i
            result[i] = be.synthetic(freq, 3)
        
        maxind = argmax(result.max((1,2)))
# WARUM IMMER MAXINDEX = 3 ???
#        print('Result Beamforming: Maxindex = ', maxind)
    
        Lm = L_p(result[maxind]).reshape(rg.shape).flatten()
    
    
        max_idx = argmax(Lm.flatten()) # position in grid with max source strength
        max_cartcoord = rg.gpos[:,max_idx]

        max_idx = argmax(Lm.flatten()) # position in grid with max source strength
        max_value = amax(Lm.flatten())
        temp_azi = arctan2(sin(rg.phi[max_idx]),cos(rg.phi[max_idx]))
        temp_ele = pi/2 - rg.theta[max_idx]
        
        max_polcoord = [rad2deg(temp_azi),
                        rad2deg(temp_ele)]

    
#### 3D-Plot ###
        # fig = plt.figure()
        # ax = fig.add_subplot(121, projection='3d')
        # ax.set_xlabel('x-Achse')
        # ax.set_ylabel('y-Achse')
        # ax.set_zlabel('z-Achse')
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)
        # cmhot = plt.get_cmap("hot_r")
        # if frame == STARTFRAME:
        #     cax = ax.scatter(x, y, -z, s=50, c=Lm, cmap=cmhot, marker='.')
        # # Mikros
        # ax.scatter(mg.mpos[0],mg.mpos[1],mg.mpos[2],'o')
        # ax.scatter(max_cartcoord[0], max_cartcoord[1], max_cartcoord[2], s=60,c='blue')
### 3D-Plot Ende ###
    
### 2D-Map ###
        # ax2 = fig.add_subplot(122)
        # ax2.set_xticks(arange(-180, 270, step=90))
        # ax2.set_yticks(arange(-90, 135, step=45))
        # cax2 = ax2.imshow(Lm.reshape(rg.shape).T, cmap=cmhot,
        #                   vmin=Lm.max()-6, vmax=Lm.max(),
        #                   extent=[-180,180,-90,90])
        # ax2.plot(max_polcoord[0],max_polcoord[1], 'bo')
    
        # fig.set_figheight(4)
        # fig.set_figwidth(8)
        # fig.colorbar(cax2)
        # fig.tight_layout(pad=3.0)
        # show()
### 2D-Map Ende ###
        
        DL_Matrix[:,:,freq_index,frame_index] = Lm.reshape(rg.shape).T
        # DL_Matrix[:,:,freq_index] = Lm.reshape(rg.shape).T
        summed_azi_x += cos(temp_azi)
        summed_azi_y += sin(temp_azi)
        summed_ele += max_polcoord[1]
        summed_max_v += max_value
        print(max_polcoord,max_value)


h5file = open_file('fold1_room2_mix002_ov1.h5', mode='w', title='Audio Data')
h5file.create_array('/', 'audio_features', obj=DL_Matrix,shape=DL_Matrix.shape)
h5file.close()


# with open('fold1_room2_mix002_ov1.npy', 'ab') as file:
#     save(file, DL_Matrix)

# with open('fold1_room2_mix002_ov1.npz') as file:
#     savez(file, DL_Matrix)

rounded_azi = arctan2(summed_azi_y,summed_azi_x)
print('rounded Azimuth: ', rad2deg(rounded_azi))
print('Rounded Elevation: ', summed_ele/FRAMES)
print('Rounded Max-Value: ', summed_max_v/FRAMES)

# # plot microphone geometry
# fig = plt.figure(2)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(mg.mpos[0],mg.mpos[1],mg.mpos[2],'o')
# show()

# =============================================================================
# plot Time Signal of maximum grid point
# =============================================================================


# max_idx = argmax(Lm.flatten()) # position in grid with max source strength

# # get beamformer (spatial filtered) output
# bfgenerator = bf.result(NUM)
# bf_output = next(bfgenerator) 

# # Erzeugt graphische Darstellung des Signals für ein Frame
# figure(3)
# title("spatial filtered signal")
# plot(bf_output[:,max_idx])
# show()

# figure(4)
# title("microphone signal of first channel")
# plot(ts.data[:NUM,0])
# show()

