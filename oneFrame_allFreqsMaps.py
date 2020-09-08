#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os import scandir, path, remove, listdir
from wavimport import WavSamples
from acoular import L_p, PowerSpectra, BeamformerEig, h5cache
from fbeam_prep import fbeampreparation
from numpy import argmax,amax,amin,arange,zeros,rad2deg,loadtxt,where
import matplotlib.pyplot as plt
from parameter import NUM, FREQBANDS, PLOTBEAMMAPS, CSV_DIR,\
    NPOINTS_ELE, NPOINTS_AZI, SINGLE_FILE_TEST

'''
3 Optionen für die Darstellung der Farben:
    1. Für jedes Fenster eigene Skalierung
    2. Für Quelle1 eine Skalierung & Quelle2 eine andere Skalierung
    3. Für alle eine Skalierung

'''

FRAME = 275

# Use Folder with 1 Audiofile
WAVFILE = listdir(SINGLE_FILE_TEST)[0]
WAVDIR = SINGLE_FILE_TEST + '/' + WAVFILE

mg, rg, st, _, _ = fbeampreparation()


ts = WavSamples(name=WAVDIR,start=FRAME*NUM,stop=(FRAME+1)*NUM)
ps = PowerSpectra(time_data=ts, block_size=512, window='Hanning')
be = BeamformerEig(freq_data=ps, steer=st, r_diag=True, n=3)

csvpath = CSV_DIR+WAVFILE[:-4]+".csv"
csvdata = loadtxt(open(csvpath, "rb"), dtype="int32", delimiter=",", usecols=(0,3,4))
fr_index = where(csvdata[:,0]==FRAME)[0][0]
if csvdata[fr_index+1,0] != FRAME:
    print("Im ausgewählten Frame ist nur 1 Quelle aktiv (nicht 2).")
    sys.exit()
Src1_AziEle = csvdata[fr_index,1:3]
Src2_AziEle = csvdata[fr_index+1,1:3]

Src1_Matrix = zeros((NPOINTS_ELE, NPOINTS_AZI, len(FREQBANDS)), dtype="float32")
Src2_Matrix = zeros((NPOINTS_ELE, NPOINTS_AZI, len(FREQBANDS)), dtype="float32")

# Opt 1
maxval1 = zeros(len(FREQBANDS))
maxval2 = zeros(len(FREQBANDS))

# Opt 2
tot_maxval1 = 0
tot_maxval2 = 0

# Opt 3
tot_maxval = 0

# Befüllen von Src1_Matrix und Src2_Matrix
for freq_index, freq in enumerate(FREQBANDS):
    
    be.n = -1 #Eigenwerte der Größe nach sortiert -> größter Eigenwert (default)
    Lm = L_p(be.synthetic(freq, 3)).reshape(rg.shape).flatten()
    Src1_Matrix[:,:,freq_index] = Lm.reshape(rg.shape).T

    max_idx1 = argmax(Lm.flatten()) # position in grid with max source strength
    max_value1 = amax(Lm.flatten())


    be.n = -2 #Eigenwerte der Größe nach sortiert -> größter Eigenwert (default)
    Lm = L_p(be.synthetic(freq, 3)).reshape(rg.shape).flatten()
    Src2_Matrix[:,:,freq_index] = Lm.reshape(rg.shape).T

    max_idx2 = argmax(Lm.flatten()) # position in grid with max source strength
    max_value2 = amax(Lm.flatten())
    
    # Opt 1
    maxval1[freq_index] = max_value1
    maxval2[freq_index] = max_value2
    
    # Opt 2
    tot_maxval1 = max([max_value1, tot_maxval])
    tot_maxval2 = max([max_value2, tot_maxval])
    
    # temp_azi = arctan2(sin(rg.phi[max_idx]),cos(rg.phi[max_idx]))
    # temp_ele = pi/2 - rg.theta[max_idx]
    
    # max_polcoord = [rad2deg(temp_azi),
    #                 rad2deg(temp_ele)]


fig, axs = plt.subplots(nrows=len(FREQBANDS), ncols=2, sharex=True, sharey=True, figsize=(6,30))
fig.tight_layout(pad=3.0)

cmhot = plt.get_cmap("hot_r")

for ind, ax in enumerate(axs.flat):
    ax.label_outer()
    ax.set_xticks(arange(-180,270, step=90))
    ax.set_yticks(arange(-90,180, step=90))
    if ind % 2 == 0:
        cax1 = ax.imshow(Src1_Matrix[:,:,int(ind/2)], cmap=cmhot,
                  vmin=maxval1[int(ind/2)]-6,
                  vmax=maxval1[int(ind/2)],
                  extent=[-180,180,-90,90])
        ax.plot(Src1_AziEle[0],Src1_AziEle[1], 'bx')
        fig.colorbar(cax1, ax=ax)
    if ind % 2 == 1:
        cax2 = ax.imshow(Src2_Matrix[:,:,int(ind/2-1)], cmap=cmhot,
                  vmin=maxval2[int(ind/2-1)]-6,
                  vmax=maxval2[int(ind/2-1)],
                  extent=[-180,180,-90,90])
        ax.plot(Src2_AziEle[0],Src2_AziEle[1], 'bx')
        fig.colorbar(cax2, ax=ax)


plt.show()
    
    # fig = plt.figure()
    # ax2 = fig.add_subplot(122)
    # ax2.set_xticks(arange(-180, 270, step=90))
    # ax2.set_yticks(arange(-90, 135, step=45))
    # cax2 = ax2.imshow(Lm.reshape(rg.shape).T, cmap=cmhot,
    #                   vmin=Lm.max()-6,
    #                   vmax=Lm.max(),
    #                   extent=[-180,180,-90,90])
    # ax2.plot(max_polcoord[0],max_polcoord[1], 'bo')

    # fig.colorbar(cax2)
    # fig.tight_layout(pad=3.0)
    # fig.suptitle(WAVFILE+' '+str(FRAME)+' '+str(freq))
    # plt.show()


if path.exists(h5cache.cache_dir+"/"+WAVFILE[:-4]+"_cache.h5"):
    remove(h5cache.cache_dir+"/"+WAVFILE[:-4]+"_cache.h5")


