#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:55:16 2018

@author: adamkujawski
"""

import os
import sys

# imports from SQL folder
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..',"SQL"))
from dfg_defsqla01 import get_source_ids, get_grid, get_source
from sql_ak_py3 import create_table, fill_table, connect_adamsnn, \
connect_arraydfg, fetchall, sq, fetcharray,fetchone, read_columns

# other imports
from acoular import RectGrid, L_p, integrate, WNoiseGenerator,PointSource,Mixer
from numpy import zeros, nonzero, array, float32, loads
from scipy.spatial import distance
from matplotlib.pyplot import figure, imshow, colorbar,show, subplot, plot,\
title



# Function returns acoular Rect Grid object from SQL Dataset
def get_acoular_grid(conn, grid_id, ap):
    '''
    function returns acoular grid from given grid id and aperture value
    '''
    
    grid_type, x1_min, x1_max, x2_min, x2_max, x3_min, dx1 = read_columns(conn=conn,
                  table_name='grid',
                  column_names=['type','x1_min','x1_max','x2_min','x2_max','x3_min','dx1'],
                  condition_column_name=['id'],
                  condition_value=[grid_id]) 
    if grid_type == 'rect':     
        acoular_grid = RectGrid(x_min=float(x1_min)*ap, x_max=float(x1_max)*ap, 
                     y_min=float(x2_min)*ap, y_max=float(x2_max)*ap, 
                     z=float(x3_min)*ap, 
                     increment=float(dx1)*ap)
    return acoular_grid


def get_target_values(conn, dataset_name, configdata_id, He):
    '''
    returns source positions and source strength of a given source distribution
    from configdata_id
    x,y,z, p2_theoretic, nsources
    '''
    # get names of config tables belonging to dataset
    configdata_name = read_columns(conn=conn,
                     table_name ='datasets',
                     column_names = ['configdata_name'],
                     condition_column_name=['dataset_name'],
                     condition_value=[dataset_name])
    configdata_name = configdata_name[0]
   # get nsources and sources_id from configprocess
    nsources, sources_id, mpos_id = read_columns(conn=conn,
                                     table_name=configdata_name,
                                     column_names=['nsources','sources_id','mpos_id'],
                                     condition_column_name=['id'],
                                     condition_value=[configdata_id])
    # get positions, p_rms and signal_id from sources
    signal_id, x1, x2, x3, p_rms = read_columns(conn=conn,
                                             table_name='sources',
                                             column_names=['signal_id', 'x1','x2','x3','p_rms'],
                                             condition_column_name=['id'],
                                             condition_value=[sources_id])
    # get aperture value
    ap = read_columns(conn=conn,
                      table_name='mpos',
                      column_names=['aperture'],
                      condition_column_name=['id'],
                      condition_value=[mpos_id])     
    # calculate p2_theoretic
    if nsources ==1:
        # get p2 from signal for calculation of p2 theoretic in maps
        p2 = read_columns(conn=conn,
                          table_name='Signal_Levels_band3',
                          column_names=['p2'],
                          condition_column_name=['signal_id','He_exp'],
                          condition_value=[signal_id,He])
        r2 = (x1*x1+x2*x2+x3*x3)*ap*ap
        p2_correct = (p2*p_rms*p_rms/r2)[0]        
    elif nsources > 1:
        p2_correct = []
        for i in range(nsources):
            # get p2 from signal for calculation of p2 theoretic in maps
            p2 = read_columns(conn=conn,
                              table_name='Signal_Levels_band3',
                              column_names=['p2'],
                              condition_column_name=['signal_id','He_exp'],
                              condition_value=[signal_id[i],He])
            r2 = (x1[i]*x1[i]+x2[i]*x2[i]+x3[i]*x3[i])*ap*ap
            p2_correct.append((p2*p_rms[i]*p_rms[i]/r2)[0])
        p2_correct = array(p2_correct)
            
    return x1,x2,x3, p2_correct, nsources

def get_dirtymap(conn,dataset_name,configdata_id,configprocess_id,He):
    # get names of config tables belonging to dataset
    dirtymap = read_columns(conn=conn,
                         table_name = dataset_name,
                         column_names = ['dirtymap'],
                         condition_column_name=['configdata_id','configprocess_id','He'],
                         condition_value=[configdata_id,configprocess_id,He])
    return loads(dirtymap)


def get_target_grid(conn,dataset_name,configdata_id,configprocess_id,He, print_target=False):
    
    # get source values 
    x1,x2,x3, p2, nsources = get_target_values(conn,dataset_name,configdata_id,He)
    
    # get names of config tables belonging to dataset
    configprocess_name = read_columns(conn=conn,
                         table_name ='datasets',
                         column_names = ['configprocess_name'],
                         condition_column_name=['dataset_name'],
                         condition_value=[dataset_name])
    configprocess_name = configprocess_name[0]
    
    # get grid id and mpos id
    grid_id, mpos_id = read_columns(conn=conn,
                                     table_name=configprocess_name,
                                     column_names=['grid_id', 'mpos_id'],
                                     condition_column_name=['id'],
                                     condition_value=[configprocess_id])    
    # get aperture value
    ap = read_columns(conn=conn,
                      table_name='mpos',
                      column_names=['aperture'],
                      condition_column_name=['id'],
                      condition_value=[mpos_id])
    
    ap = ap[0]     
    # get acoular grid
    grid = get_acoular_grid(conn, grid_id, ap)
    
    # build map
    if nsources == 1:
        target_map = zeros((grid.nxsteps,grid.nysteps),dtype=float32) # initialize map with zeros
        grid_index = grid.index(x1*ap, x2*ap) 
        target_map[grid_index[0],grid_index[1]] = p2

    if print_target:
        extent = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
        print(extent)
        figure(1)
        imshow( L_p(target_map).T, origin='lower',vmax=L_p(target_map.max()),vmin=L_p(target_map.max())-10,extent=extent)
        plot(x1*ap,x2*ap,'rx')
        colorbar()  
        show()
        
    return target_map, grid_index, p2, x1, x2, x3, nsources

#
#def create_target(cursor, grid, sources_id, fftdata_id, sampling_id, He_exp):
#    target_map = zeros((grid.nxsteps,grid.nysteps),dtype=float32) # initialize map with zeros
#    source_ids = fetchall(cursor, get_source_ids, sources_id) # all ids of given sources
#    cursor.execute(sq(get_ap,1))
#    (ap) = cursor.fetchone()
#
#    p2_theoretic = [] # each theoretical source strength of all sources
#
#    for source_id in source_ids:
#        cursor.execute(sq(get_source_parameters, sources_id, source_id))# get the source parameters one after the other 
#        x1, x2, x3, signal_id, p_rms = cursor.fetchone()                
#        r2 = (x1*x1+x2*x2+x3*x3)*ap[0]*ap[0]
#        p2_correct = fetchone(cursor, get_Sig_p2,
#                            signal_id, 
#                            fftdata_id, 
#                            sampling_id,
#                            He_exp)*p_rms*p_rms/r2
#        grid_index = grid.index(x1*ap[0], x2*ap[0])
#        target_map[grid_index[0],grid_index[1]] = p2_correct
#
#        p2_theoretic.append(p2_correct)   
#
#    return p2_theoretic, target_map



#def sector(x1,x2,ap, r_def):
#    #r_def -> defined radius of circle normed on aparture
#    sec_size = sector_size(x1,x2,ap,r_def)         
#    sector = (x1*ap,x2*ap,sec_size)
#    return sector


def sector_size(x1,x2,ap,r_def):
    # x1, x2 -> type list
    x1 = array(x1)*ap
    x2 = array(x2)*ap 

    r_bw = r_def*ap
    if x1.shape[0] > 1:
        loc = list(zip(x1,x2)) # zip x and y values together
        dist = distance.cdist(loc,loc,'euclidean').flatten() # callculate all distances from all sources
        r_min = min(dist[nonzero(dist)]) # find smallest distance between sources 
        sec_size = min(r_min/2,r_bw)
    else:
        sec_size = r_bw
    return sec_size                           
    



def get_sql_sources(cursor,sources_id,sfreq,mpos):
    # sources_id -> id für eine Quellkartierung mit bestimmten Quellen
    # source_id -> id jeder einzelnen source
    # source ids -> liste aller ids der einzelnen sources einer sources id 
    source_ids = fetchall(cursor, get_source_ids, sources_id)
    cursor.execute(sq(get_ap,1))
    (ap) = cursor.fetchone()
    sourcelist = [] # list     
    for source_id in source_ids:
        cursor.execute(sq(get_source, sources_id, source_id))         # get the source parameters one after the other
        (signal_id, 
         x1, x2, x3, 
         pol_type,
         dipole_id, 
         p_rms ) = cursor.fetchone()                
        # generate harmonic or noise source...
        sgnl = WNoiseGenerator(rms=p_rms, 
                               sample_freq=sfreq, 
                               numsamples=512000, 
                               seed=(signal_id-1))        
    #    newsrc = PointSource(signal = sgnl, 
    #                         mpos = m_error, 
    #                         loc = (x1*ap, x2*ap, x3*ap))        
        newsrc = PointSource(signal = sgnl, 
                             mpos = mpos, 
                             loc = (x1*ap[0], x2*ap[0], x3*ap[0]))                                   
        sourcelist.append(newsrc) # add the new source to the list    
    if len(source_ids) > 1: # if there are multiple sources, they have to be mixed
        src = Mixer(source = sourcelist[0],sources = sourcelist[1:])    
    else: # if there's only one source, it is the only one in the list
        src = sourcelist[0]
    return src


get_ap = '''
SELECT aperture
FROM  mpos
WHERE id = %s
'''

get_p2_theoretic = '''
SELECT p2_theoretic
FROM Task_Levels
WHERE He_exp = %s
AND Tasks_id = %s
AND source_id = %s 
'''
get_source_parameters = '''
SELECT x1, x2, x3, signal_id, p_rms 
FROM sources
WHERE id = %s
AND source_id = %s 
'''

get_Sig_p2 = '''
SELECT p2
FROM Signal_Levels_band3
WHERE signal_id = %s
AND fftdata_id = %s
AND sampling_id = %s
AND He_exp = %s
''' 
##############################################################################


if __name__ == "__main__":
    conn, curs = connect_adamsnn()    
    get_target_grid(conn,'dirtymap_1src',7087,5,0,print_target=True)    
    
