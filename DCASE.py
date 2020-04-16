#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:55:01 2020

@author: pablo
"""

from os import path, remove, pardir
import acoular
import numpy as np
from scipy.io import wavfile
import tables
import matplotlib.pyplot as plt
from pylab import figure, subplot, plot, axis, imshow, colorbar, show, xlabel, ylabel, title, tight_layout
from mpl_toolkits.mplot3d import Axes3D

micgeofile = path.join(path.split(acoular.__file__)[0],'xml','DCASE20_Tetraeder.xml')
h5savefile = '/home/pablo/Dokumente/Uni/Bachelorarbeit/SELD_DCASE2020/DCASE_Test.h5'
try:
    remove(h5savefile)
except OSError:
    pass
#read data from wav
samplefreq, wavdata = wavfile.read('/home/pablo/Downloads/foa_dev/fold1_room1_mix001_ov1.wav', mmap=False)


#octave band of interest
cfreq = 4000

# ANORDNUNG DER MIKROS
mg = acoular.MicGeom( from_file=micgeofile )

# MESSBEREICH
rg = acoular.RectGrid3D(x_min=-2, x_max=2,
                        y_min=-2, y_max=2,
                        z_min=-2, z_max=2,
                        increment=0.05 )



#output
acoularh5 = tables.open_file(h5savefile, mode = "w")
acoularh5.create_earray('/','time_data', atom=None, title='', filters=None, \
                         chunkshape=[256,64], \
                         byteorder=None, createparents=False, obj=wavdata)
acoularh5.set_node_attr('/time_data','sample_freq', samplefreq)
acoularh5.close()


# Zeitsamples
t1 = acoular.MaskedTimeSamples(name=h5savefile)
t1.start = 0 # first sample, default
t1.stop = 16000 # last valid sample = 15999

st = acoular.SteeringVector(grid=rg, mics=mg)


fi = acoular.FiltFiltOctave(source=t1, band=cfreq)
bts = acoular.BeamformerTimeSq(source=fi, steer=st, r_diag=True)
avgts = acoular.TimeAverage(source=bts, naverage = 1024)
cachts = acoular.TimeCache( source = avgts) # cache to prevent recalculation

i1 = 1
i2 = 1 # no of figure
# first, plot time-dependent result (block-wise)
figure(i2,(7,7))
i2 += 1
res = np.zeros(rg.size) # init accumulator for average
i3 = 1 # no of subplot
for r in cachts.result(1):  #one single block
    subplot(4,4,i3)
    i3 += 1
    res += r[0] # average accum.
    map = r[0].reshape(rg.shape)
    mx = acoular.L_p(map.max())
    imshow(acoular.L_p(map.T), vmax=mx, vmin=mx-15, origin='lower',
           interpolation='nearest', extent=rg.extend())
    title('%i' % ((i3-1)*1024))
res /= i3-1 # average
tight_layout()
# second, plot overall result (average over all blocks)
figure(1)
subplot(4,4,i1)
i1 += 1
map = res.reshape(rg.shape)
mx = acoular.L_p(map.max())
imshow(acoular.L_p(map.T), vmax=mx, vmin=mx-15, origin='lower',
       interpolation='nearest', extent=rg.extend())
colorbar()
title(('BeamformerTime','BeamformerTimeSq')[i2-3])
tight_layout()


###############################################################################
###############################################################################
# analyze the data
#ts = acoular.TimeSamples( name=h5savefile )
#ps = acoular.PowerSpectra(time_data=ts,
#                          block_size=256,
#                          window='Hamming',
#                          #overlap='87.5%',
#                          #ind_low=5, ind_high=16,
#                          cached=True) #default
#st = acoular.SteeringVector(grid=rg, mics=mg, steer_type='true location')
#bb = acoular.BeamformerBase( freq_data=ps, steer=st )



# Plotten der Mikros
#fig = plt.figure(2)
#ax = Axes3D(fig)
#ax.plot(mg.mpos[0],mg.mpos[1],mg.mpos[2], 'o')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#ax.set_xlim(-10, 10)
#ax.set_ylim(-10, 10)
#ax.set_zlim(-10, 10)

