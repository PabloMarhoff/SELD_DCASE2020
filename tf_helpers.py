#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf


def _float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_TFRecord(fname: str, feature_pipeline, verbosity:int=100):
    """
    writes standard TFRecord file that can be consumed by tensorflows 
    tf.data API. Stores data in binary strings.    
    
    TFRecords can be tf.SequenceExample or tf.trainExample.This funtion is
    only valid for tf.trainExample (non-Sequential data).
    
    Parameters
    ----------
    fname : str
        filename of the TFrecord.
    feature_pipeline : python generator
        generator that yields the features to be written iteratively. Need to be
        as tf.Example compatible mapping of type {"string": tf.train.Feature}.
    verbosity : int, optional
        prints write process for a defined interval of samples. The default is 100.

    Returns
    -------
    None.

    """

    with tf.io.TFRecordWriter(fname) as writer:
        for feature in feature_pipeline:
            # wrap data as TF Features
            data = tf.train.Features(feature=feature)
            # wrap again as a TF Example
            example = tf.train.Example(features=data)
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        writer.close()
