#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduced Version of ResNetDfg.py. Adjusted to tensorflow version > 2.0
For full migration: https://www.tensorflow.org/guide/migrate

Created on Thu May 05 16:43:41 2020

@author: kujawski

"""

import sys
import os
#import tensorflow as tf
from time import time
import shutil
import numpy as np
import multiprocessing
import pandas as pd

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import Tensorflow models
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..',"tf_models"))
from adamsnn_CNN_models import LocalizationEstimationModelConstructor,\
ResNetOfficial, TimeHistory
# other imports 
from folder_structure import create_session_folders
### decorator function
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..',"Dataset_Reader"))
from dataset_decorators import tensorflow_input_fn

# =============================================================================
# parameters
# =============================================================================

# task params
task = 'None'
BATCHSIZE = 8 # number of training samples per batch
LRATE = 0.001 # learning rate used to update model weights

#training params:
SAMPLES_PER_EPOCH = 16000 # 16000 samples are used to train until the model is tested
ITERATIONS = SAMPLES_PER_EPOCH//BATCHSIZE # number of training iterations per epoch
EPOCHS = 150 # 150: total number of Training epochs
TESTSTEPS = None #None

# Model Params
SEED = 10 # random seed 
INGRIDSIZE = 51*51 # size of the input image (source map)
NXSTEPS = NYSTEPS = 51
AVAILABLE_CPUs = multiprocessing.cpu_count() # number of CPUs used for efficient input pipeline

# datasets
TRAINRECORD = [f'TFrecord_files/dm_training_1src_HeExpAll_Grid{INGRIDSIZE}_ch1_normalized.tfrecords']  
TESTRECORD = [f'TFrecord_files/dm_test_1src_HeExpAll_Grid{INGRIDSIZE}_ch1_normalized.tfrecords']  

# folder structure: creates a session folder including folder is with saved model  
CALCDIR,SESSIONDIR,BESTDIR = create_session_folders(os.path.split(__file__)[0],
                                   "ResNet_grid{}".format(INGRIDSIZE),
                                   'task',task)
log_folder = os.path.split(SESSIONDIR)[-1]

# =============================================================================
# Input Pipeline
# =============================================================================

# input function
def create_input_fn(TFrecord_filenames, batch_size, shuffle, seed=None):
    @tensorflow_input_fn(TFrecord_filenames, batch_size, shuffle, seed=SEED, parallel_calls = AVAILABLE_CPUs)
                         
    def parser(record):
        keys_to_features = {'dirtymap': tf.VarLenFeature(tf.float32),
            'p2': tf.FixedLenFeature((), tf.float32),
            'coordinates' : tf.VarLenFeature(tf.float32)}
        parsed = tf.parse_single_example(record, keys_to_features)
    
        # Perform additional preprocessing on the parsed data.
        dm = tf.reshape(tf.sparse_tensor_to_dense(parsed['dirtymap']),[NXSTEPS,NYSTEPS,1]) # maybe direct from byte string map
        coordinates = tf.reshape(tf.sparse_tensor_to_dense(parsed['coordinates']),[2,1])
        p2 = tf.reshape(parsed["p2"],[1,1])
        return {"features": dm}, {"labels": tf.squeeze(tf.concat([coordinates,p2],0)), 
               'coordinates': tf.squeeze(coordinates),
               'p2':tf.squeeze(p2)}
    return parser

# =============================================================================
# Model
# =============================================================================

time_hist = TimeHistory()

ResNet = ResNetOfficial(numhiddenlayers=0,
                        average_pooling=True,
                        inputgridsize=INGRIDSIZE,
                        random_seed = SEED,
        output_shape = {'1': [-1,3]}, 
          output_fc_nodes = {'1':3})
                          
opt = tf.train.AdamOptimizer(LRATE)
#opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=len(cluster['worker']),
#total_num_replicas=len(cluster['worker']))  
#sync_replicas_hook = opt.make_session_run_hook(is_chief)

# neural network processor
pnn = LocalizationEstimationModelConstructor(network = ResNet,
                           optimizer = opt,
                           output_target_mapper = {'1':'labels'})

#strategy = tf.contrib.distribute.MirroredStrategy(
#                                                  devices=devices
#                                                  )

# Estimator Config
config = tf.estimator.RunConfig(model_dir=SESSIONDIR,
                                save_summary_steps=100,
                                keep_checkpoint_max = 5,
                                log_step_count_steps = 100,
#                                train_distribute=strategy,
#                                eval_distribute=strategy
                                )
# Create the Estimator
estimator = tf.estimator.Estimator(config=config,model_fn=pnn.model_func,model_dir=SESSIONDIR)
   
# =============================================================================
#Training
# =============================================================================
t = time()
losses = []
gs = []

# Input Funcs  
training_input_fn = create_input_fn(TRAINRECORD, BATCHSIZE, shuffle=True,seed=SEED)
test_input_fn = create_input_fn(TESTRECORD, BATCHSIZE, shuffle=False)

for _ in range(EPOCHS):
    estimator.train(input_fn=training_input_fn,steps=ITERATIONS,hooks=[time_hist])  
    test_results = estimator.evaluate(input_fn=test_input_fn, steps=TESTSTEPS)   
    losses.append(test_results['loss'])
    gs.append(test_results['global_step'])
    
    if test_results['loss'] <= min(losses): #
        calc_time_minimum = time()-t
        total_training_time_minimum = sum(time_hist.times)
        # move checkpoint files in best results folder:
        filenames = [name for name in os.listdir(SESSIONDIR) if str(test_results['global_step']) in name]
        for name in filenames: shutil.copy(os.path.join(SESSIONDIR,name), os.path.join(BESTDIR,name))

    total_training_time = sum(time_hist.times)
    calc_time = time()-t
    avg_time_per_batch = np.mean(time_hist.times)
    images_per_second = BATCHSIZE/avg_time_per_batch

    print(f"total time with {AVAILABLE_CPUs} CPU(s): {total_training_time} seconds")
    print(f"{BATCHSIZE/avg_time_per_batch} images/second with {AVAILABLE_CPUs} CPU(s)")        

    COLUMNS = ['log_folder','time','time_minimum','training_time','img_per_sec','loss','global_step']
#    SET loss = %s, log_folder = '%s', time = %s, time_minimum = %s,training_time=%s,img_per_sec=%s, global_step = %s
    sql_table_data = dict(zip(COLUMNS, [
                                  log_folder,
                                  calc_time,
                                  calc_time_minimum,
                                  total_training_time_minimum,
                                  images_per_second, 
                                  min(losses),
                                  gs[np.argmin(losses)]]))
    df = pd.DataFrame(sql_table_data, index=[0])
    df.to_csv(SESSIONDIR + r'/model_results.csv',sep='\t', encoding='utf-8')

##            
##    tr = {'global_step': global_steps, 'loss':losses, 'distance_error': distance_error, 
##          'localization_accuracy': accuracy, 'source_level_error':level_error}
##    df = pandas.DataFrame.from_dict(tr)  
##    df.to_csv(SESSIONDIR + r'/test_results.csv',sep='\t', encoding='utf-8')


