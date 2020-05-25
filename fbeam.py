#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from acoular import __file__ as bpath, MicGeom, WriteH5, PowerSpectra, SteeringVector,\
 BeamformerBase, L_p, BeamformerTime, TimeAverage, TimePower,\
 FiltFiltOctave, WriteWAV, BeamformerClean, BeamformerTimeSq, BeamformerEig, BeamformerOrth
from pylab import figure, plot, axis, show, title, zeros, deg2rad,rad2deg,array
from wavimport import WavSamples
from sphericalGrids import SphericalGrid_Equiangular
from numpy import argmax,amax,pi,arange,arctan2,sin,cos,arccos,sqrt,zeros
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from parameter import STARTFRAME, ENDFRAME, NUM, NPOINTS_AZI, NPOINTS_ELE, FREQBANDS

def fbeamextraction(filepath, debug=False, detailedinfo_lvl=1, plotbeammaps=False):
## TODO nicht bei jeder Datei neu ausführen!
    # Gradangaben von Theta im Intervall [0,180] statt wie bei DCASE [90,-90]
    M1 = spherical2cart(deg2rad(45),deg2rad(55),0.042)
    M2 = spherical2cart(deg2rad(315),deg2rad(125),0.042)
    M3 = spherical2cart(deg2rad(135),deg2rad(125),0.042)
    M4 = spherical2cart(deg2rad(225),deg2rad(55),0.042)
    mg = MicGeom()
    mg.mpos_tot = array([M1,M2,M3,M4]).T # add microphone positions to MicGeom object
##
    
    # define evaluation grid
    rg = SphericalGrid_Equiangular(NPOINTS_AZI, NPOINTS_ELE)
       
    
    x = rg.gpos[0]
    y = rg.gpos[1]
    z = rg.gpos[2]
    
    if debug or plotbeammaps or detailedinfo_lvl >= 2:
        summed_azi_x = 0.0
        summed_azi_y = 0.0
        summed_ele = 0.0
        summed_max_v = 0.0
        # TODO print verschieben
        # print('[Azimuth, Elevation], max_value')
        firstframe = STARTFRAME
        lastframe = ENDFRAME
        FRAMES = ENDFRAME - STARTFRAME
        ts = WavSamples(name=filepath,start=firstframe*NUM,stop=(firstframe+1)*NUM)
    else:
        firstframe = 0
        lastframe = 600
        FRAMES = 600
        ts = WavSamples(name=filepath,start=0,stop=NUM) # first Frame from File
    
    st = SteeringVector(grid=rg, mics=mg)
    ps = PowerSpectra(time_data=ts, block_size=512, window='Hanning')
    be = BeamformerEig(freq_data=ps, steer=st, r_diag=True, n=3)
    # bb = BeamformerBase(freq_data=ps, steer=st)
    # bo = BeamformerOrth(beamformer=be, eva_list=[3])
    
    ##################### DeepLearning-Matrix ##############################
    ################   Elevation    Azimuth      Frequenzbänder  Frames  ###
    DL_Matrix = zeros((NPOINTS_ELE, NPOINTS_AZI, len(FREQBANDS), FRAMES)) ##
    ########################################################################    
    
    for frame_index, frame in enumerate(range(firstframe, lastframe)):
        if detailedinfo_lvl >= 1:
            print('  FRAME: ', frame-firstframe, ' (',frame,')')
        for freq_index, freq in enumerate(FREQBANDS):
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
            DL_Matrix[:,:,freq_index,frame_index] = Lm.reshape(rg.shape).T

            if detailedinfo_lvl >= 2 or plotbeammaps:
                max_idx = argmax(Lm.flatten()) # position in grid with max source strength
                max_value = amax(Lm.flatten())

                max_cartcoord = rg.gpos[:,max_idx]
        
                temp_azi = arctan2(sin(rg.phi[max_idx]),cos(rg.phi[max_idx]))
                temp_ele = pi/2 - rg.theta[max_idx]
                
                max_polcoord = [rad2deg(temp_azi),
                                rad2deg(temp_ele)]
                summed_azi_x += cos(temp_azi)
                summed_azi_y += sin(temp_azi)
                summed_ele += max_polcoord[1]
                summed_max_v += max_value
            if detailedinfo_lvl >= 3:
                print('   ',freq,max_polcoord,max_value)
    
        
    #### 3D-Plot ###
            if plotbeammaps:
                fig = plt.figure()
                ax = fig.add_subplot(121, projection='3d')
                ax.set_xlabel('x-Achse')
                ax.set_ylabel('y-Achse')
                ax.set_zlabel('z-Achse')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                cmhot = plt.get_cmap("hot_r")
                if frame == firstframe:
                    ax.scatter(x, y, -z, s=50, c=Lm, cmap=cmhot, marker='.')
                # Mikros
                ax.scatter(mg.mpos[0],mg.mpos[1],mg.mpos[2],'o')
                ax.scatter(max_cartcoord[0], max_cartcoord[1], max_cartcoord[2], s=60,c='blue')
    ### 3D-Plot Ende ###
        
    ### 2D-Map ###
                ax2 = fig.add_subplot(122)
                ax2.set_xticks(arange(-180, 270, step=90))
                ax2.set_yticks(arange(-90, 135, step=45))
                cax2 = ax2.imshow(Lm.reshape(rg.shape).T, cmap=cmhot,
                                  vmin=Lm.max()-6, vmax=Lm.max(),
                                  extent=[-180,180,-90,90])
                ax2.plot(max_polcoord[0],max_polcoord[1], 'bo')
            
                fig.set_figheight(4)
                fig.set_figwidth(8)
                fig.colorbar(cax2)
                fig.tight_layout(pad=3.0)
                show()
    ### 2D-Map Ende ###
            
    
    if detailedinfo_lvl >= 2:
        rounded_azi = arctan2(summed_azi_y,summed_azi_x)
        print('     => Rounded Azimuth: ', rad2deg(rounded_azi))
        print('     => Rounded Elevation: ', summed_ele/FRAMES)
        print('     => Rounded Max-Value: ', summed_max_v/FRAMES)
    
    return DL_Matrix

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


