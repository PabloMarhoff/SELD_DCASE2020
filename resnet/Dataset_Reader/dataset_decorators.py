#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:19:41 2018

@author: adamkujawski
"""

import tensorflow as tf
if not tf.__version__[0] == '1':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()  
else:
    pass

def tensorflow_input_fn(TFrecord_filenames, batch_size, shuffle=True, seed=None,
                        buffer_size = 1000,parallel_calls=None):

    def _tensorflow_input_fn(func):
        def func_wrapper():
            dataset = tf.data.TFRecordDataset(TFrecord_filenames)
            dataset = dataset.map(func,num_parallel_calls=parallel_calls)#
            if shuffle:
                dataset = dataset.shuffle(buffer_size=buffer_size,seed=seed)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE) # has been before: tf.contrib.data.AUTOTUNE
#            dataset = dataset.make_one_shot_iterator()
#            features, labels = dataset.get_next()
#            return features, labels
            return dataset
    
        return func_wrapper
    return _tensorflow_input_fn
