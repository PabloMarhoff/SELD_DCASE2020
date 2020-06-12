#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from acoular import L_p
from numpy import argmax,amax,pi,arange,arctan2,sin,cos,zeros,rad2deg, where
import matplotlib.pyplot as plt
from parameter import NUM, FREQBANDS, DETAILEDINFO_LVL, PLOTBEAMMAPS, NPOINTS_ELE, NPOINTS_AZI
from tf_helpers import _float_list_feature, _int64_feature
import mpl_toolkits.mplot3d
from csv_reader import csv_extractor


# Weil csvdata nicht jeden Frame enthält, sondern aktuell nur die, wo #Quellen==1, muss
# als Hilfsvariable index eingeführt werden.
# Index = Position in csvdata
# frame = Framenummer der Audiodatei
def fbeamextraction(mg, rg, ts, be, firstframe, lastframe, csvdata):

    #####################  DeepLearning-Matrix (Framewise)  #########################
    ################    Elevation     Azimuth    Frequenzbänder   ###################
    DL_Matrix = zeros((NPOINTS_ELE, NPOINTS_AZI, len(FREQBANDS)), dtype="float32") ##
    #################################################################################

    for frame in range(firstframe, lastframe):
        if frame in csvdata[:,0]:
            fr_index = where(csvdata[:,0]==frame)[0][0]
            if DETAILEDINFO_LVL >= 1:
                print('  FRAME: ', frame-firstframe, ' (',frame,')')
            ts.start = frame*NUM
            ts.stop = (frame+1)*NUM
    
            for freq_index, freq in enumerate(FREQBANDS):
    
                result = zeros((4,rg.shape[0],rg.shape[1]))
                for i in range(4):
                    be.n = i
                    result[i] = be.synthetic(freq, 3)
                maxind = argmax(result.max((1,2)))
        # WARUM IMMER MAXINDEX = 3 ???
        #        print('Result Beamforming: Maxindex = ', maxind)
            
                Lm = L_p(result[maxind]).reshape(rg.shape).flatten()
    
                # result = zeros((rg.shape[0],rg.shape[1]))
                # be.n = 3
                # result = be.synthetic(freq, 3)
                # Lm = L_p(result).reshape(rg.shape).flatten()
    
                
                DL_Matrix[:,:,freq_index] = Lm.reshape(rg.shape).T
    
                if PLOTBEAMMAPS:
                    max_idx = argmax(Lm.flatten()) # position in grid with max source strength
                    max_value = amax(Lm.flatten())
    
                    max_cartcoord = rg.gpos[:,max_idx]
            
                    temp_azi = arctan2(sin(rg.phi[max_idx]),cos(rg.phi[max_idx]))
                    temp_ele = pi/2 - rg.theta[max_idx]
                    
                    max_polcoord = [rad2deg(temp_azi),
                                    rad2deg(temp_ele)]
        
        #### 3D-Plot ###
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
                        ax.scatter(rg.gpos[0], rg.gpos[1], -rg.gpos[2], s=50, c=Lm, cmap=cmhot, marker='.')
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
                    plt.show()
        ### 2D-Map Ende ###
    
                if DETAILEDINFO_LVL >= 3:
                    print('   ',freq,max_polcoord,max_value)
    
    
        
            if DETAILEDINFO_LVL >= 2:
                print("   .csv-Values: Class  Azi  Ele")
                print("                ",
                      '{:4d}'.format(csvdata[fr_index, 1]),
                      '{:4d}'.format(csvdata[fr_index, 2]),
                      '{:4d}'.format(csvdata[fr_index, 3]))
                # print("                ",
                #       '{:4d}'.format(csvdata[frame*2+1, 1]),
                #       '{:4d}'.format(csvdata[frame*2+1, 2]),
                #       '{:4d}'.format(csvdata[frame*2+1, 3]))
                
            feature_dict = {
                'inputmap': _float_list_feature(DL_Matrix),
                'class': _int64_feature(csvdata[fr_index, 1]),
                # 'class_2': _int64_feature(csvdata[frame*2+1,1]),
                'azi':   _int64_feature(csvdata[fr_index, 2]),
                # 'azi_2':   _int64_feature(csvdata[frame*2+1,2]),
                'ele':   _int64_feature(csvdata[fr_index, 3]),
                # 'ele_2':   _int64_feature(csvdata[frame*2+1,3])
                }
            yield feature_dict
