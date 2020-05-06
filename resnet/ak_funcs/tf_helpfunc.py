#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:05:28 2018

@author: adamkujawski
"""


import tensorflow as tf
import numpy as np
#
def argmax_2d(tensor):
    '''
    returns position index of maximum value in 2d array/ 2d map
    '''
  # input format: BxHxWxD
    #  assert len(tensor.get_shape()) == 4
      # flatten the Tensor along the height and width axes
    flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))
      # argmax of the flat tensor
    argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
      # convert indexes into 2D coordinates
    argmax_x = argmax // tf.shape(tensor)[2]
    argmax_y = argmax % tf.shape(tensor)[2]
      # stack and return 2D coordinates
    return tf.stack((argmax_x, argmax_y), axis=1)

def log10(x):
    '''
    at the moment no log10 in tensorflow implemented..
    building log10 in tensorflow -> https://github.com/tensorflow/tensorflow/issues/1666 
    '''
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def tf_L_p(x):  
    p0 = 4e-10
    arg = tf.div(x,p0)    
    arg_clip = tf.clip_by_value(arg,1e-35,10**20) # maximum is unrealistic because none type is not allowed
    return 10*log10(arg_clip)





