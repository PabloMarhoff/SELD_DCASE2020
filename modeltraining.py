#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from parameter import FEATURE_DIR, NPOINTS_AZI, NPOINTS_ELE, FREQBANDS
import multiprocessing
from time import time
import shutil
import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




# import Tensorflow models
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"resnet","tf_models"))
from adamsnn_CNN_models import LocalizationEstimationModelConstructor,\
ResNetOfficial, TimeHistory
# other imports 
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"resnet","runscripts"))
from folder_structure import create_session_folders
### decorator function
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"resnet","Dataset_Reader"))
from dataset_decorators import tensorflow_input_fn



#%% TODO
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
INGRIDSIZE = NPOINTS_AZI*NPOINTS_ELE # size of the input image (source map)
NXSTEPS = NPOINTS_AZI
NYSTEPS = NPOINTS_ELE
AVAILABLE_CPUs = multiprocessing.cpu_count() # number of CPUs used for efficient input pipeline

# datasets
TRAINRECORD = FEATURE_DIR # Trennung der 3,4,5,6
TESTRECORD = FEATURE_DIR  # Datensätze   1,2
# folder structure: creates a session folder including folder is with saved model  
CALCDIR,SESSIONDIR,BESTDIR = create_session_folders(os.path.split(__file__)[0],
                                   "ResNet_grid{}".format(INGRIDSIZE),
                                   'task',task)
log_folder = os.path.split(SESSIONDIR)[-1]


#%% PARSER der Inputdaten aus den .tfrecord-Dateien

# =============================================================================
# Input Pipeline
# =============================================================================

# TODO: map_func mit create_input_fn (aus dataset_decorators.py) umschließen?!?
# input function
def create_input_fn(TFrecord_filenames, batch_size, shuffle, seed=None):
    @tensorflow_input_fn(TFrecord_filenames, batch_size, shuffle, seed=SEED, parallel_calls = AVAILABLE_CPUs)
    def map_func(record):
        feature_description = {
            'inputmap': tf.io.VarLenFeature(tf.float32),
            'class': tf.io.FixedLenFeature([], tf.int64),
            # 'class_2': ...
            'azi':   tf.io.FixedLenFeature([], tf.int64),
            # 'azi_2': ...
            'ele':   tf.io.FixedLenFeature([], tf.int64),
            # 'ele_2': ...
            }
        # Parse the input tf.Example proto using the dictionary above.
        parsed = tf.io.parse_single_example(record, feature_description)
        # Perform additional preprocessing on the parsed data.
        # reshape the inputmap again after parsing the list of floats
        inputmap = tf.reshape(tf.sparse.to_dense(parsed['inputmap']),[NPOINTS_ELE, NPOINTS_AZI, len(FREQBANDS)])
        inputclass = parsed['class']
        azi = parsed['azi'] # TODO in Radiant umrechnen, dann auf Wert zwischen 0..1
        ele = parsed['ele'] # TODO same
        # returns {features}, {labels} as tuple! Inputclass entfernt
        return {"features": inputmap}, {"labels":tf.stack([azi,ele],0),
                                        'azi':azi,
                                        'ele':ele}
    return map_func

#%% TODO: output_shape & output_fc_nodes
# =============================================================================
# Model
# =============================================================================

time_hist = TimeHistory()


ResNet = ResNetOfficial(numhiddenlayers=0,
                        average_pooling=True,
                        inputgridsize=INGRIDSIZE,
                        random_seed = SEED,
        output_shape = {'1': [-1,2]}, 
          output_fc_nodes = {'1':2})
                          
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

#%%

# =============================================================================
#Training
# =============================================================================
t = time()
losses = []
gs = []

# Input Funcs
testfile = ['/home/pablo/Dokumente/Uni/Bachelorarbeit/SELD_DCASE2020/extracted_features/fold1_room1_mix001_ov1.tfrecords']
#testdataset = 
training_input_fn = create_input_fn(testfile, BATCHSIZE, shuffle=True,seed=SEED)# TRAINRECORD, BATCHSIZE, shuffle=True,seed=SEED)
print("TRAINING: ", training_input_fn)
test_input_fn = create_input_fn(testfile, BATCHSIZE, shuffle=False)#TESTRECORD, BATCHSIZE, shuffle=False)
print("TEST: ", test_input_fn)

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


#%% TESTFUNKTION für Parser
with os.scandir(FEATURE_DIR) as files:
    for file in files:
        raw_dataset = tf.data.TFRecordDataset(file.path)
        dataset = raw_dataset.map(map_func)
        # beispielshalber alle Frames durchgehen
        for i,(features,labels) in enumerate(dataset):
            if file.name == 'fold1_room2_mix026_ov2.tfrecords':
                #print (i, labels['azi'].numpy(), labels['ele'].numpy())
                pass
