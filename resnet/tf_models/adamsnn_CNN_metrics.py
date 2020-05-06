#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 17:41:04 2018

@author: adamkujawski
"""

import tensorflow as tf
import os
import sys
#imports from ak_funcs folder 
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..',"ak_funcs")) 
from tf_helpfunc import tf_L_p, argmax_2d


def distance_error(coordinates, coordinates_prediction, axis=1):
    '''
    simply the L2Norm as a calculation of distance between true 
    source position and predicted source position.
    Works at the moment just for single sources maps

    Inputs: 
    -------
    coordinates -> x1,x2 coordinates in shape [batchsize, 2]

    Returns:
    -------
    l2diff -> distance in shape [batchsize]
    '''
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(coordinates, coordinates_prediction)),
                                reduction_indices=axis)) 

def distance_error_grid(coordinates,map_prediction,dx1=0.02):
    '''
    returns mean position error distance between predicted source and 
    true source position
    '''
    argmax_pred = tf.squeeze(tf.cast(argmax_2d(map_prediction),tf.float32),axis=2)  #shape = (batchsize,xiyi, 1) 
    def xy_pos(x):
        _x1 = (x[0])*dx1-0.5
        _x2 = (x[1])*dx1-0.5
        return _x1, _x2
    _x1, _x2 = tf.map_fn(lambda x: xy_pos(x), argmax_pred, dtype=(tf.float32,tf.float32))    
    coordinates_prediction = tf.stack([_x1,_x2],axis=1) #shape = (batchsize,x1x2, 1)

    return tf.sqrt( tf.reduce_sum(tf.square(tf.subtract(coordinates, coordinates_prediction)),
                                reduction_indices=1)) 
    

def localization_accuracy(coordinates, coordinates_prediction, d=0.05):
    '''
    Inputs: 
    -------
    - coordinates -> x1,x2 coordinates in shape [batchsize, 2]
    - d -> size of circle in which a predicted source is correctly identified 

    Returns:
    -------
    - '1' if prediction lay in circle area, otherwise '0' (output shape = [batchsize])
    '''

    l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(coordinates, coordinates_prediction)),
                                reduction_indices=1))  
    in_circle = tf.less(l2diff,d)           
    accuracy = tf.cast(in_circle,dtype=tf.float32)

    return accuracy

def localization_accuracy_grid(coordinates,map_prediction,dx1=0.02,d=0.05):

    argmax_pred = tf.squeeze(tf.cast(argmax_2d(map_prediction),tf.float32),axis=2)  #shape = (batchsize,xiyi, 1) 
    def xy_pos(x):
        _x1 = (x[0])*dx1-0.5
        _x2 = (x[1])*dx1-0.5
        return _x1, _x2
    _x1, _x2 = tf.map_fn(lambda x: xy_pos(x), argmax_pred, dtype=(tf.float32,tf.float32))    
    coordinates_prediction = tf.stack([_x1,_x2],axis=1) #shape = (batchsize,x1x2, 1)
    l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(coordinates, coordinates_prediction)),
                                reduction_indices=1))  

    in_circle = tf.less(l2diff,d)           
    accuracy = tf.cast(in_circle,dtype=tf.float32)

    return accuracy

def source_level_error(p2, p2_prediction):
    '''
    Inputs:
    ------
    - p2(shape=[batchsize,1])
    
    Returns:
    - positiv Level difference (shape=[batchsize,1])
    '''
    return tf.abs(tf_L_p(p2) - tf_L_p(p2_prediction))

def source_level_error_grid(p2, map_prediction):
    '''
    Inputs:
    ------
    - p2(shape=[batchsize,1])
    
    Returns:
    - positiv Level difference (shape=[batchsize,1])
    '''
    
    p2_prediction = tf.reduce_max(map_prediction,axis=[1,2])    
    
    return tf.abs(tf_L_p(p2) - tf_L_p(p2_prediction))

def q_error_grid(p2, map_prediction):
    '''
    Inputs:
    ------
    - p2(shape=[batchsize,1])
    
    Returns:
    - positiv Level difference (shape=[batchsize,1])
    '''
    
    p2_prediction = tf.reduce_max(map_prediction,axis=[1,2])    
    
    return tf.square(p2 - p2_prediction)
