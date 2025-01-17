#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
from os import listdir, scandir
from os.path import isfile, join, dirname, split
from parameter import FEATURE_DIR_TESTING_SPLIT, FEATURE_DIR_TRAINING_SPLIT,\
    NPOINTS_AZI, NPOINTS_ELE, FREQBANDS, TRAINING, PLOTFILES
import multiprocessing
from time import time
import shutil
from numpy import pi, mean, arctan2, argmin, zeros, save, load
import pandas as pd

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




# import Tensorflow models
sys.path.insert(0,join(dirname(__file__),"resnet","tf_models"))
from adamsnn_CNN_models import LocalizationEstimationModelConstructor,\
ResNetOfficial, TimeHistory
# other imports 
sys.path.insert(0,join(dirname(__file__),"resnet","runscripts"))
from folder_structure import create_session_folders
### decorator function
sys.path.insert(0,join(dirname(__file__),"resnet","Dataset_Reader"))
from dataset_decorators import tensorflow_input_fn



#%% PARAMETERS

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

# datasets (Datapaths as a List)
TRAINRECORD = [f for f in listdir(FEATURE_DIR_TRAINING_SPLIT) if isfile(join(FEATURE_DIR_TRAINING_SPLIT, f))]
TRAINRECORD = [FEATURE_DIR_TRAINING_SPLIT + e for e in TRAINRECORD]
TESTRECORD = [f for f in listdir(FEATURE_DIR_TESTING_SPLIT) if isfile(join(FEATURE_DIR_TESTING_SPLIT, f))]
TESTRECORD = [FEATURE_DIR_TESTING_SPLIT + e for e in TESTRECORD]
# folder structure: creates a session folder including folder is with saved model  
CALCDIR,SESSIONDIR,BESTDIR = create_session_folders(split(__file__)[0],
                                   "ResNet_grid{}".format(INGRIDSIZE),
                                   'task',task)
log_folder = split(SESSIONDIR)[-1]

# Ohne Thresholdfilter
#CKPT = "/home/pablo/Dokumente/Uni/Bachelorarbeit/SELD_DCASE2020/calc_out_ResNet_grid2450/16_16_58_best_model/best_results/model.ckpt-300000"

# Mit Threshold, aber "falscher" Ele_Error-Formel
#CKPT = "/home/pablo/Dokumente/Uni/Bachelorarbeit/SELD_DCASE2020/calc_out_ResNet_grid2450/17_35_07_best_model_thresholdfilter/best_results/model.ckpt-188000"

# Finales Training
CKPT = "/home/pablo/Dokumente/Uni/Bachelorarbeit/SELD_DCASE2020/calc_out_ResNet_grid2450/20200825_Threshold_Ele/best_results/model.ckpt-296000"


#%% PARSER der Inputdaten aus den .tfrecord-Dateien

# =============================================================================
# Input Pipeline
# =============================================================================

# input function
def create_input_fn(TFrecord_filenames, batch_size, shuffle, seed=None):
    @tensorflow_input_fn(TFrecord_filenames, batch_size, shuffle, seed=SEED, parallel_calls = AVAILABLE_CPUs)
    def map_func(record):
        if TRAINING:
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
            normed_input = tf.sparse.to_dense(parsed['inputmap'])/tf.math.reduce_max(tf.sparse.to_dense(parsed['inputmap']))
            # NICHT-NORMIERTE VERSION: normed_input = tf.sparse.to_dense(parsed['inputmap'])
            normed_inputmap = tf.reshape(normed_input,[NPOINTS_ELE, NPOINTS_AZI, len(FREQBANDS)])
            inputclass = parsed['class']
            azi = parsed['azi']
            ele = parsed['ele']
            azi_sin = tf.math.sin(tf.cast(azi, tf.float32) * pi/180)
            azi_cos = tf.math.cos(tf.cast(azi, tf.float32) * pi/180)
            ele_sin = tf.math.sin(tf.cast(ele, tf.float32) * pi/180)
            ele_cos = tf.math.cos(tf.cast(ele, tf.float32) * pi/180)
    # returns {features}, {labels} as tuple!
            return {"features": normed_inputmap}, {"labels":tf.stack([azi_sin,azi_cos,ele_sin,ele_cos],0),
                                                   'azi_sin':azi_sin,
                                                   'azi_cos':azi_cos,
                                                   'ele_sin':ele_sin,
                                                   'ele_cos':ele_cos}
        if not(TRAINING):
            feature_description = {
                'inputmap': tf.io.VarLenFeature(tf.float32),
                }
            # Parse the input tf.Example proto using the dictionary above.
            parsed = tf.io.parse_single_example(record, feature_description)
            
            # Perform additional preprocessing on the parsed data.
            # reshape the inputmap again after parsing the list of floats
            normed_input = tf.sparse.to_dense(parsed['inputmap'])/tf.math.reduce_max(tf.sparse.to_dense(parsed['inputmap']))
            # NICHT-NORMIERTE VERSION: normed_input = tf.sparse.to_dense(parsed['inputmap'])
            normed_inputmap = tf.reshape(normed_input,[NPOINTS_ELE, NPOINTS_AZI, len(FREQBANDS)])
    # returns {features}, {labels} as tuple!
            return {"features": normed_inputmap}
            
    return map_func




#%% Config
# =============================================================================
# Model
# =============================================================================

time_hist = TimeHistory()


ResNet = ResNetOfficial(numhiddenlayers=0,
                        average_pooling=True,
                        inputgridsize=INGRIDSIZE,
                        random_seed = SEED,
                        output_shape = {'1': [-1,4]}, 
                        output_fc_nodes = {'1':4})
                          
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

#%% TRAINING

t = time()
losses = []
gs = []

# Input Funcs
if TRAINING:
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
            filenames = [name for name in listdir(SESSIONDIR) if str(test_results['global_step']) in name]
            for name in filenames: shutil.copy(join(SESSIONDIR,name), join(BESTDIR,name))
    
        total_training_time = sum(time_hist.times)
        calc_time = time()-t
        avg_time_per_batch = mean(time_hist.times)
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
                                      gs[argmin(losses)]]))
        df = pd.DataFrame(sql_table_data, index=[0])
        df.to_csv(SESSIONDIR + r'/model_results.csv',sep='\t', encoding='utf-8')


#%% PREDICTING

# ACHTUNG: INPUT MUSS AUCH NORMIERT SEIN!! --> passiert in create_input_fn
if not(TRAINING):
    with scandir(FEATURE_DIR_TESTING_SPLIT) as files:
        for file in files:
            # um alle Audiodateien getrennt auszuwerten, wird in jedem Durchlauf eine Liste
            # mit nur einer Datei angelegt
            PREDRECORD = [file.path]
            # print(PREDRECORD)
            # create_input_fn(Dateiliste, Batchgröße, Inputframes mischen?)
            prediction_input = create_input_fn(PREDRECORD, 1000, shuffle=False)
            prediction_generator = estimator.predict(input_fn=prediction_input, yield_single_examples=False,checkpoint_path = CKPT)
            npyfile = PLOTFILES+file.name[:-10]
            activeframes = load(npyfile+'_frames.npy')
            try:
                pred = next(prediction_generator)
                aziEle_rad = [arctan2(pred['1'][:,0],pred['1'][:,1]), arctan2(pred['1'][:,2], pred['1'][:,3])]
                frAziEle_deg = zeros((aziEle_rad[0].size, 3))
                #frAziEle_deg[:,0] = activeframes[:,0] weil Fehlermeldung "too many indices for array"
                frAziEle_deg[:,0] = activeframes[:]
                frAziEle_deg[:,1] = aziEle_rad[0] * 180/pi
                frAziEle_deg[:,2] = aziEle_rad[1] * 180/pi
                save(npyfile+'_KNN.npy',frAziEle_deg)
                # remove(npyfile+'_frames.npy')
            except StopIteration:
                pass

    
