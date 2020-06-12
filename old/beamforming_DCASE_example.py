#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:40:09 2020

@author: kujawski

"""

from os import path
from acoular import __file__ as bpath, MicGeom, WNoiseGenerator, PointSource,\
 Mixer, WriteH5, TimeSamples, PowerSpectra, RectGrid, SteeringVector,\
 BeamformerBase, L_p,SineGenerator, BeamformerTime, TimeAverage, TimePower,\
 FiltFiltOctave, WriteWAV, BeamformerClean, BeamformerTimeSq
from pylab import figure, plot, axis, imshow, colorbar, show,title, zeros,\
    deg2rad,rad2deg,array
from wavimport import WavSamples, SphericalGrid_Equiangular, SphericalGrid_EvenlyDist
from numpy import argmax,amax,pi,arange,arctan2,sin,cos,arccos,sqrt

# 100 ms Frames -> fs = 24000    
NUM = 2400 #samples per frame
AUDIO_DIR = "/home/pablo/Dokumente/Uni/Bachelorarbeit/SELD_DCASE2020/TAU-NIGENS/mic_dev/"
TRACK = "fold1_room2_mix002_ov1.wav"


# Anzahl Stützstellen für Beamforming (AZI:ELE = 2:1)
NPOINTS_AZI = 70
NPOINTS_ELE = 35

# Zu analysierender Bereich der WAV-Datei
STARTFRAME = 350
ENDFRAME = 370
# Anzahl der zu analysierenden Frames (<ENDFRAME-STARTFRAME)
FRAMES = 10



# =============================================================================
# Data Processing

# Das Processing funktioniert in Funktionsblöcken:
# Zeitdaten(TimeSamples) -> räumliche Filterung(BeamformerTime) -> \
# spektrale Filterung(FiltFiltOctave) -> Energie(TimePower,TimeAverage)    

# man könnte hier auch noch einen Bandpass Filter Block dazwischenschalten,
# der existiert aber noch nicht.
# =============================================================================

# create Microphone Array from DCASE
import math
# phi : azimuth
# theta : elevation
def spherical2cart(phi,theta,r):
    return array([
         r * sin(theta) * cos(phi),
         r * sin(theta) * sin(phi),
         r * cos(theta)
    ])
def cart2spherical(x,y,z):
    return array([
        arctan2(y,x) * 180/pi,
        arccos(z/(sqrt(x**2+y**2+z**2))) * 180/pi
    ])
def cart2spherical_dcase(x,y,z):
    phi = arctan2(y,x) * 180/pi
    theta = arccos(z/(sqrt(x**2+y**2+z**2))) * 180/pi
    return array([phi,90-theta])


# Gradangaben von Theta im Intervall [0,180] statt wie bei DCASE [90,-90]
M1 = spherical2cart(deg2rad(45),deg2rad(55),0.042)
M2 = spherical2cart(deg2rad(315),deg2rad(125),0.042)
M3 = spherical2cart(deg2rad(135),deg2rad(125),0.042)
M4 = spherical2cart(deg2rad(225),deg2rad(55),0.042)

mg = MicGeom()
mg.mpos_tot = array([M1,M2,M3,M4]).T # add microphone positions to MicGeom object

# define evaluation grid
# Hier könntest du vielleicht eine neue Spherical Grid Klasse schreiben oder 
# eine ArbitraryGrid Klasse, damit wir ein sinnvolles Gitter zur Lokalisierung
# verwenden können.
# Als Anregung siehe: https://spaudiopy.readthedocs.io/en/latest/spaudiopy.grids.html
#
rg = SphericalGrid_Equiangular(NPOINTS_AZI, NPOINTS_ELE)
st = SteeringVector(grid=rg, mics=mg)
    
# analyze the data and generate map
name = AUDIO_DIR+TRACK
ts = WavSamples(name=name,start=STARTFRAME*NUM,stop=ENDFRAME*NUM)
bf = BeamformerTime(source=ts,steer=st)
#ft = FiltFiltOctave(source=bf,band=4000)
tp = TimePower(source=bf)
tavg = TimeAverage(source=tp,naverage=NUM)

# =============================================================================
# plot first data block
# =============================================================================
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

gen = tavg.result(1) # make generator object

## show map
#imshow( Lm.T, origin='lower', vmin=Lm.max()-10, extent=rg.extend, \
#interpolation='bicubic')
#colorbar()

# # hier muss man sich jetzt überlegen, wie man am Besten die sphärischen
# Datenpunkte plottet. Scatter ist eine Variante

x = rg.gpos[0]
y = rg.gpos[1]
z = rg.gpos[2]

summed_azi_x = 0.0
summed_azi_y = 0.0
summed_ele = 0.0
summed_max_v = 0.0
print('[Azimuth, Elevation], max_value')
for frame in range(0, FRAMES):
    Lm = L_p(next(gen)).reshape(rg.shape) # get next block from generator pipeline
    max_idx = argmax(Lm.flatten()) # position in grid with max source strength
    max_cartcoord = rg.gpos[:,max_idx]

    fig = plt.figure(frame)
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel('x-Achse')
    ax.set_ylabel('y-Achse')
    ax.set_zlabel('z-Achse')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    cmhot = plt.get_cmap("hot_r")

    if frame == 0:
        cax = ax.scatter(x, y, -z, s=50, c=Lm, cmap=cmhot, marker='.')
  
    ax.scatter(mg.mpos[0],mg.mpos[1],mg.mpos[2],'o')
    ax.scatter(max_cartcoord[0], max_cartcoord[1], max_cartcoord[2], s=60,c='blue')


    ax2 = fig.add_subplot(122)
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-100, 100)
    ax2.set_xticks(arange(-180, 270, step=90))
    ax2.set_yticks(arange(-90, 135, step=45))

    cmhot = plt.get_cmap("hot_r")
    max_idx = argmax(Lm.flatten()) # position in grid with max source strength
    max_value = amax(Lm.flatten())
    temp_azi = arctan2(sin(rg.phi[max_idx]),cos(rg.phi[max_idx]))
    temp_ele = pi/2 - rg.theta[max_idx]
    
    max_polcoord = [rad2deg(temp_azi),
                    rad2deg(temp_ele)]
    cax = ax2.scatter(rad2deg(arctan2(sin(rg.phi),cos(rg.phi))),
                      rad2deg(pi/2 - rg.theta),
                      s=50, c=Lm, cmap=cmhot, marker='.')
    ax2.scatter(max_polcoord[0], max_polcoord[1], s=60,c='blue')
    fig.set_figheight(4)
    fig.set_figwidth(8)
    fig.colorbar(cax)
    fig.tight_layout(pad=3.0)
    show()
    
    summed_azi_x += cos(temp_azi)
    summed_azi_y += sin(temp_azi)
    summed_ele += max_polcoord[1]
    summed_max_v += max_value

    print(frame,': ', max_polcoord,max_value)


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
