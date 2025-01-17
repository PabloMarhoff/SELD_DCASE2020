#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from acoular import L_p
from numpy import argmax,amax,amin,pi,arange,arctan2,sin,cos,zeros,rad2deg,where,save,array,sum,delete
import matplotlib.pyplot as plt
from parameter import NUM, FREQBANDS, DETAILEDINFO_LVL, PLOTBEAMMAPS,\
    NPOINTS_ELE, NPOINTS_AZI, TRAINING, JUST_1SOURCE_FRAMES
from tf_helpers import _float_list_feature, _int64_feature
from mpl_toolkits.mplot3d import Axes3D



'''
Händische Bestimmung via Algo von Azi&Ele der Quelle.
1. (unabhängig von Intensität der einzelnen Freqbänder) Außreißer eliminieren
2. Übrige Winkelwerte mitteln

# IN: Maxima der Beamformingmaps für alle Freqs
# OUT: Prediction für Azi&Ele (rad)
'''
def angle_calc_algo(azis,azic,eles,elec):
    # ohne die UNTERSTEN 3 FREQBÄNDER, weil diese stark abweichen
    azis = azis[3:]
    azic = azic[3:]
    eles = eles[3:]
    elec = elec[3:]
    
    mean_azis = sum(azis)/len(azis)
    mean_azic = sum(azic)/len(azic)
    mean_eles = sum(eles)/len(eles)
    mean_elec = sum(elec)/len(elec)
    
    # LÖSCHEN DER BEIDEN GRÖßTEN WERTE anhand Array aus den
    # absoluten Werten der Abweichungen vom Mittelwert
    azis_temp = list(abs(azis - mean_azis))
    del azis[argmax(azis_temp)]
    del azis_temp[argmax(azis_temp)]
    del azis[argmax(azis_temp)]
    
    azic_temp = list(abs(azic - mean_azic))
    del azic[argmax(azic_temp)]
    del azic_temp[argmax(azic_temp)]
    del azic[argmax(azic_temp)]
    
    eles_temp = list(abs(eles - mean_eles))
    del eles[argmax(eles_temp)]
    del eles_temp[argmax(eles_temp)]
    del eles[argmax(eles_temp)]
    
    elec_temp = list(abs(elec - mean_elec))
    del elec[argmax(elec_temp)]
    del elec_temp[argmax(elec_temp)]
    del elec[argmax(elec_temp)]
    
    # engültige Mittelwerte und Prediction von Azi & Ele
    mean_azis = sum(azis)/len(azis)
    mean_azic = sum(azic)/len(azic)
    mean_eles = sum(eles)/len(eles)
    mean_elec = sum(elec)/len(elec)
    
    azi_pred = arctan2(mean_azis,mean_azic)
    ele_pred = arctan2(mean_eles,mean_elec)
    return azi_pred, ele_pred



#%%
# Weil csvdata nicht jeden Frame enthält, sondern nur aktive Frames (aktuell
# nur die, wo #Quellen==1), wird fr_index als Hilfsvariable eingeführt.
# fr_index = Position in csvdata
# frame = Framenummer der Audiodatei
# BEI PREDICTION: csvdata enthält Frame-Liste mit Frames energiereicher als Threshold
def fbeamextraction(mg, rg, ts, be, firstframe, lastframe, csvdata, _name, fbeamplot, fbeamplot_2ndSrc, algoplot, algoplot_2ndSrc):

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
            
            # Zur Berechnung des Mittelwerts der Richtungswinkel
            if not(TRAINING):
                azis = list(zeros(len(FREQBANDS)))
                azic = list(zeros(len(FREQBANDS)))
                eles = list(zeros(len(FREQBANDS)))
                elec = list(zeros(len(FREQBANDS)))
                azis_2nd = list(zeros(len(FREQBANDS)))
                azic_2nd = list(zeros(len(FREQBANDS)))
                eles_2nd = list(zeros(len(FREQBANDS)))
                elec_2nd = list(zeros(len(FREQBANDS)))
            
#%%  1. Quelle  ##############################
            # Zur Angle-Prediction ohne händischem Algo
            glob_maxval = 0
            glob_maxidx = 0
            for freq_index, freq in enumerate(FREQBANDS):
                
                be.n = -1 #Eigenwerte sind der Größe nach sortiert! -> größter Eigenwert (default)
                Lm = L_p(be.synthetic(freq, 3)).reshape(rg.shape).flatten()
                
                DL_Matrix[:,:,freq_index] = Lm.reshape(rg.shape).T
                
                if PLOTBEAMMAPS or DETAILEDINFO_LVL>=3 or not(TRAINING):
                    max_idx = argmax(Lm.flatten()) # position in grid with max source strength
                    max_value = amax(Lm.flatten())
                    if glob_maxval < max_value: # TODO: 'and freq != 500:' !?!
                        glob_maxidx = max_idx
                        glob_maxval = max_value
                    # min_value = amin(Lm.flatten())
    
                    max_cartcoord = rg.gpos[:,max_idx]
            
                    temp_azi = arctan2(sin(rg.phi[max_idx]),cos(rg.phi[max_idx]))
                    temp_ele = pi/2 - rg.theta[max_idx]
                    
                    max_polcoord = [rad2deg(temp_azi),
                                    rad2deg(temp_ele)]
                    
                    azis[freq_index] = sin(temp_azi)
                    azic[freq_index] = cos(temp_azi)
                    eles[freq_index] = sin(temp_ele)
                    elec[freq_index] = cos(temp_ele)
                    
                if PLOTBEAMMAPS:
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
                                      vmin=Lm.max()-6,
                                      vmax=Lm.max(),
                                      extent=[-180,180,-90,90])
                    ax2.plot(max_polcoord[0],max_polcoord[1], 'bo')
                
                    fig.set_figheight(4)
                    fig.set_figwidth(8)
                    fig.colorbar(cax2)
                    fig.tight_layout(pad=3.0)
                    fig.suptitle(_name+' '+str(frame)+' '+str(freq))
                    plt.show()
        ### 2D-Map Ende ###
    
                if DETAILEDINFO_LVL >= 3:
                    print('   {:4d} [{:7.2f}, {:7.2f}] {}'.format(freq,round(max_polcoord[0],2),round(max_polcoord[1],2),round(max_value,2)))
                    #print('   ',freq, max_cartcoord, max_value)
                    
        
            if DETAILEDINFO_LVL >= 2 and TRAINING:
                print("   .csv-Values: Class  Azi  Ele")
                print("                ",
                      '{:4d}'.format(csvdata[fr_index, 1]),
                      '{:4d}'.format(csvdata[fr_index, 2]),
                      '{:4d}'.format(csvdata[fr_index, 3]))

            
            if TRAINING:
                feature_dict = {
                    'inputmap': _float_list_feature(DL_Matrix),
                    'class': _int64_feature(csvdata[fr_index, 1]),
                    'azi':   _int64_feature(csvdata[fr_index, 2]),
                    'ele':   _int64_feature(csvdata[fr_index, 3]),
                    }
            if not(TRAINING):
                # Prediction nur übers globale Maximum
                azi_pred = arctan2(sin(rg.phi[glob_maxidx]),cos(rg.phi[glob_maxidx]))
                ele_pred = pi/2 - rg.theta[glob_maxidx]
                fbeamplot.append([frame, rad2deg(azi_pred), rad2deg(ele_pred)])

                # Calculation per Algorithmus (basierend auf Ergebnissen des fbeamformings)
                azi_algo, ele_algo = angle_calc_algo(azis,azic,eles,elec)
                algoplot.append([frame, rad2deg(azi_algo), rad2deg(ele_algo)])

                feature_dict = {
                    'inputmap': _float_list_feature(DL_Matrix)
                    }
            
            yield feature_dict


#%%  2. Quelle  ##############################
            # Nur, wenn nicht bereits am Ende des Arrays (wenn bei letztem aktiven Frame nur 1 Quelle)
            if not(fr_index+1==len(csvdata)):
            
                if not(JUST_1SOURCE_FRAMES or TRAINING) and (frame==csvdata[fr_index+1,0]):
                    glob_maxval = 0
                    glob_maxidx = 0
                    
                    if DETAILEDINFO_LVL >= 1:
                        print('  FRAME: ', frame-firstframe, ' (',frame,'), 2nd Src')
    
                    for freq_index, freq in enumerate(FREQBANDS):
                    
                        be.n = -2 # zweitgrößter Eigenwert -> zweitstärkste Quelle
                        Lm = L_p(be.synthetic(freq,3)).reshape(rg.shape).flatten()
                        DL_Matrix[:,:,freq_index] = Lm.reshape(rg.shape).T
                        
                        
                        max_idx = argmax(Lm.flatten()) # position in grid with max source strength
                        max_value = amax(Lm.flatten())
                        if glob_maxval < max_value: # TODO: 'and freq != 500:' !?!
                            glob_maxidx = max_idx
                            glob_maxval = max_value
                        
                        max_cartcoord = rg.gpos[:,max_idx]
                        
                        temp_azi = arctan2(sin(rg.phi[max_idx]),cos(rg.phi[max_idx]))
                        temp_ele = pi/2 - rg.theta[max_idx]
                            
                        max_polcoord = [rad2deg(temp_azi),
                                        rad2deg(temp_ele)]
                        
                        azis_2nd[freq_index] = sin(temp_azi)
                        azic_2nd[freq_index] = cos(temp_azi)
                        eles_2nd[freq_index] = sin(temp_ele)
                        elec_2nd[freq_index] = cos(temp_ele)
                        
                        
                        if PLOTBEAMMAPS:
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
                            fig.suptitle(_name+' (Frame:'+str(frame)+', Freq:'+str(freq)+') 2nd Source')
                            plt.show()
                ### 2D-Map Ende ###
                            
                        if DETAILEDINFO_LVL >= 3:
                            print('   {:4d} [{:7.2f}, {:7.2f}] {} (2nd Src)'.format(freq,round(max_polcoord[0],2),round(max_polcoord[1],2),round(max_value,2)))
                            #print('   ',freq, max_cartcoord, max_value)
        
            
            
                    # Prediction nur übers globale Maximum
                    azi_pred_2nd = arctan2(sin(rg.phi[glob_maxidx]),cos(rg.phi[glob_maxidx]))
                    ele_pred_2nd = pi/2 - rg.theta[glob_maxidx]
                    fbeamplot_2ndSrc.append([frame, rad2deg(azi_pred_2nd), rad2deg(ele_pred_2nd)])

                    # Calculation per Algorithmus (basierend auf Ergebnissen des fbeamformings)
                    azi_algo_2nd, ele_algo_2nd = angle_calc_algo(azis_2nd,azic_2nd,eles_2nd,elec_2nd)
                    algoplot_2ndSrc.append([frame, rad2deg(azi_algo_2nd), rad2deg(ele_algo_2nd)])
                    
                    feature_dict = {
                        'inputmap': _float_list_feature(DL_Matrix)
                        }
                    
                    yield feature_dict



