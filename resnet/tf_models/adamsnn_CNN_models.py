#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:57:03 2018

@author: adamkujawski

when using multiple losses this should be the correct way:
    https://stackoverflow.com/questions/49953379/tensorflow-multiple-loss-functions-vs-multiple-training-ops

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
if not tf.__version__[0] == '1':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()  
else:
    pass


from traits.api import HasTraits, List, Tuple,Trait, Str, Int, ListInt, \
ListStr, Instance, Dict, HasPrivateTraits, Property, Bool, Float, Any
import sys
import os
import numpy as np

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..',"tf_models/models/official/resnet"))
import resnet_model
from adamsnn_CNN_metrics import distance_error, localization_accuracy, source_level_error,\
 localization_accuracy_grid, distance_error_grid, source_level_error_grid,q_error_grid
import time


class ResNetClass(resnet_model.Model):
  """Model class with uses CIFAR10 structure for maps with size 51x51.
     
  resnet_size should be either 20, 32, 44, 56, ... 
  """

  def __init__(self, resnet_size, inputgridsize, average_pooling_type, data_format='channels_last',
               resnet_version=2,
               dtype=tf.float32,
               average_pooling=True,
               random_seed=None):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
      to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.

    Raises:
      ValueError: if invalid resnet_size is chosen
    """
    
    tf.set_random_seed(random_seed)

    if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    num_filters = int(np.round(np.sqrt(inputgridsize)/2))
    final_size = num_filters
    for _ in range(2): final_size = final_size*2
    
    super(ResNetClass, self).__init__(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=None,
        num_filters=num_filters,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        final_size=final_size,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype,
        average_pooling=average_pooling,
        average_pooling_type = average_pooling_type
    )

class ResNetOfficial(HasPrivateTraits):
    
    avg_pooling_type = Trait(1,2) # 1 -> pooling like in He et al, 2 -> pooling along channels dimension
    
    inputgridsize = Int(2601)

    numhiddenlayers = Int(1)
    
    numhiddennodes = Int(256)

    average_pooling = Bool(True)

    resnet_size = Int(20) # 20, 32, 44, 56 layers...
    
    random_seed = Any(None)

    output_shape = Dict({'1': [-1],
                         '2': [-1,2]}, 
            desc='individual shape of each output')
    
    output_fc_nodes = Dict({'1':1,
                            '2':2},
            desc = 'nodes of individual last fullyconnected layer of output')    

    def network_constructor(self, inputs, mode):
        """
        function constructs convolutional neural network from class
        information
        returns all layers as list and output object which provides data
        """
        print("construct network...")

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        print("is_training:",is_training)

        print("initialized grid size: ", self.inputgridsize)

        model = ResNetClass(resnet_size=self.resnet_size,
                            inputgridsize=self.inputgridsize,
                            average_pooling=self.average_pooling,
                            average_pooling_type = self.avg_pooling_type,
                            random_seed = self.random_seed) 
        
        print("final number of filters: ",model.final_size)
    
        resnet_output = model(inputs, mode == tf.estimator.ModeKeys.TRAIN)

        # add multiple outputs with dense layer
        out = dict()        
        
        for output in list(self.output_shape.keys()):
#            with tf.name_scope("dense_output{}".format(output)): # layer name scope
            inputs = tf.identity(resnet_output, 'resnet_output_{}'.format(output))
            for i in range(self.numhiddenlayers):
                inputs = tf.layers.dense(inputs=inputs,
                         units=self.numhiddennodes, 
                         activation=None,
                         kernel_initializer=tf.variance_scaling_initializer(
                                 scale=1.0, mode='fan_avg', distribution='uniform',
                                 seed=self.random_seed),
                         name='dense_layer_{}_{}'.format(i,output))

            final_layer_output = tf.layers.dense(inputs=inputs,
                    units=self.output_fc_nodes[output],
                    activation=None,
                    kernel_initializer=tf.variance_scaling_initializer(
                            scale=1.0, mode='fan_avg', distribution='uniform',seed=self.random_seed),
                    name="final_layer_output_"+str(output))

            # add to output dictionary
            out[output] = tf.reshape(final_layer_output, self.output_shape[output])

###        # add some information to output
#        print([n.name for n in tf.get_default_graph().as_graph_def().node])
#        relu_ops = [n.name for n in tf.get_default_graph().as_graph_def().node if 'Relu' in n.name]
#        conv2d_ops = [n.name for n in tf.get_default_graph().as_graph_def().node if 'Conv2D' in n.name]
#        regression_ops = [n.name for n in tf.get_default_graph().as_graph_def().node if 'resnet_output' in n.name or 'dense_layer' in n.name]
##        kernel_ops = [n.name for n in tf.get_default_graph().as_graph_def().node if 'kernel' in n.name and 'conv2d' in n.name]
#        for lt in [relu_ops,conv2d_ops,regression_ops]:
#            for op in lt:
#                out[op] = tf.get_default_graph().get_tensor_by_name(op+':0')
#        
#        #out['Relu'] = self.get_tensor_values(relu_ops)
##        out['Conv2D'] = self.get_tensor_values(conv2d_ops)
##        kernel_ops = [n.name for n in tf.get_default_graph().as_graph_def().node if 'kernel' in n.name and 'conv2d' in n.name]
##        out['kernels'] = self.get_tensor_values(kernel_ops)
#        
        return out 

 

class ProcessNeuralNetwork(HasPrivateTraits):
    '''
    Network defines training, test and evaluation parameters and provides
    model function for tensorflow estimator    
    '''
    network = Instance(object)
                
    optimizer = Trait()
    
    def model_func(self, features, labels, mode):

        out = self.network.network_constructor(inputs=features["features"], mode=mode)

        if mode == tf.estimator.ModeKeys.PREDICT: # if model func to predict
            return tf.estimator.EstimatorSpec(mode=mode, predictions=out)

#       # Calculate Loss (for both TRAIN and EVAL modes)       
        loss = tf.losses.mean_squared_error(labels=labels['targets'],predictions=out)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = self.optimizer
            train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
        # Add evaluation metrics (for EVAL mode)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss) 

class LocalizationEstimationModelConstructor(ProcessNeuralNetwork):
    '''
    Model function processes multiple output values
    # at the moment there is no weighting of the parameters which contribute
    to the loss function
    
    Multisource case:
    labels coordinates -> [batch_size, max_sources, 2]
    labels p2 -> [batch_size]
    
    '''

    weight_decay = Float()

    loss_scale = Float(1)
    
    max_sources = Int(1, desc="max number of sources in a map. Must match with labels.")
    
    localization_accuracy_radius = Float(0.05)
    
    #id_tracker = Any([])

    output_target_mapper = Dict({'1':'p2',
                                 '2':'coordinates'},
                desc = "maps output key to target key for loss calculation")
    
    ### need a weighting parameter for different loss values ####
# TODO Ele und Azi hinzufügen
    metric_funcs = Dict({'azi_error': True,
                         'ele_error': True,
                         'distance_error': False,
                         'source_level_error': False,
                         'localization_accuracy': False},
                        desc='additional functions for evaluation at training and testphase of network')

# TODO ähnlich wie distance_error
    def _get_metrics(self, labels, predictions):
        eval_metric_ops = dict()
        #metrics:
        if self.metric_funcs['azi_error']:
            # Betrag der Differenzen vom Sinus und Kosinus
            abs_sin = tf.math.abs(labels['azi_sin']-predictions['1'][:,0])
            abs_cos = tf.math.abs(labels['azi_cos']-predictions['1'][:,1])
            dist_error = tf.math.atan2(abs_sin,abs_cos)
            
            # hier falsch, da Daten schon normiert sind
            # dist_error = tf.linalg.norm(dist_error)
            
            mean_dist_error = tf.reduce_mean(dist_error)
            # fügt azi_error ins Dict "eval_metrics_ops" hinzu
            eval_metric_ops["azi_error"] = tf.metrics.mean(mean_dist_error) 
            # Visualisierung in TensorBoard
            tf.summary.scalar('azi_error',mean_dist_error)

        if self.metric_funcs['ele_error']:
            # dist_error = tf.linalg.norm(tf.math.atan2(labels['ele_sin'],labels['ele_cos'])-tf.math.atan2(predictions['1'][:,2],predictions['1'][:,3]),axis=0)
            dist_error = tf.math.abs(tf.math.atan2(labels['ele_sin'],labels['ele_cos'])-tf.math.atan2(predictions['1'][:,2],predictions['1'][:,3]))
            mean_dist_error = tf.reduce_mean(dist_error)
            eval_metric_ops["ele_error"] = tf.metrics.mean(mean_dist_error) 
            tf.summary.scalar('ele_error',mean_dist_error)


        if self.metric_funcs['distance_error']:
            dist_error = tf.linalg.norm(labels['coordinates']-predictions['1'][:,:2],axis=1)
            mean_dist_error = tf.reduce_mean(dist_error)
            eval_metric_ops["distance_error"] = tf.metrics.mean(mean_dist_error) 
            tf.summary.scalar('distance_error',mean_dist_error)

        if self.metric_funcs['localization_accuracy']:
            loc_acc = localization_accuracy(labels['coordinates'], predictions['2'], d=self.localization_accuracy_radius)
            eval_metric_ops["localization_accuracy"] = tf.metrics.mean(loc_acc)             
            tf.summary.scalar('localization_accuracy',tf.metrics.mean(loc_acc)[1])

        if self.metric_funcs['source_level_error']:
            source_lev_error = source_level_error(labels['p2'], predictions['1'][:,2])
            mean_source_lev_error = tf.reduce_mean(source_lev_error)
            eval_metric_ops["source_level_error"] = tf.metrics.mean(source_lev_error)          
#            tf.summary.scalar('source_level_error',tf.metrics.mean(source_lev_error)[1])
            tf.summary.scalar('source_level_error',mean_source_lev_error)
        #eval_metric_ops['loss_nowd'] = tf.metrics.mean_squared_error(labels['labels'],predictions['1'])
        return eval_metric_ops        

    def mse(self,v1,v2,axis=0):
        return tf.cast(tf.reduce_mean(tf.square(v1 - v2),axis=axis),tf.float32)

    def custom_loss_elementwise(self,nsources,label,prediction):
        """
        labs -> [nsources,(dim)]
        preds -> [dim]
        """
        mse_over_sources = tf.map_fn(lambda i: self.mse(prediction,label[i],axis=-1), 
         tf.range(0,nsources,1),dtype=tf.float32)
        #return tf.reduce_min(mse_over_sources)
        return mse_over_sources

#    def custom_loss_elementwise(self,nsources,labls_list,pred_list):
#        """
#        labs -> [nsources,(dim)]
#        preds -> [dim]
#        """
#        #loss = tf.zeros(nsources)
#        for l, p in zip(labls_list,pred_list):
##            r = tf.rank(l)
##            res = tf.map_fn(lambda i: self.mse(p,l[i],axis=r-1), 
##                 tf.range(0,nsources,1),dtype=tf.float32) 
#            res = tf.map_fn(lambda i: self.mse(p,l[i],axis=-1), 
#             tf.range(0,nsources,1),dtype=tf.float32)
#        #loss = tf.cond(tf.rank(res)>1,lambda:tf.add(loss,res[:,0]),lambda:tf.add(loss,res))
#        return tf.reduce_min(res)        
        
    def model_func(self, features, labels, mode):
        
        out = self.network.network_constructor(inputs=features["features"], mode=mode)

        if mode == tf.estimator.ModeKeys.PREDICT: # if model func to predict

#            kernel_ops = [n.name for n in tf.get_default_graph().as_graph_def().node if 'kernel' in n.name and 'conv2d' in n.name]
#            for op in kernel_ops:
#                out[op] = tf.get_default_graph().get_tensor_by_name(op+':0')            
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              predictions=out,
                                              export_outputs={'prediction': tf.estimator.export.PredictOutput(out)})

        eval_metric_ops = self._get_metrics(labels, out) 

#        if self.max_sources == 1:
       # Calculate Loss (for both TRAIN and EVAL modes) 
        loss_objects = list()
        for output_key in self.output_target_mapper.keys():
            loss = tf.losses.mean_squared_error(labels=labels[self.output_target_mapper[output_key]],predictions=out[output_key])
#            labs=labels[self.output_target_mapper[output_key]]
#            preds = out[output_key]
#            if self.output_target_mapper[output_key] == 'p2': 
#                loss = tf.reduce_mean(self.mse(labs,preds,axis=0))
#            if self.output_target_mapper[output_key] == 'coordinates': 
#                loss = tf.reduce_mean(self.mse(labs,preds,axis=1))
            loss_objects.append(loss)
        final_loss = tf.reduce_sum(loss_objects) # maybe a weightening function is needed
#        
#        elif self.max_sources > 1:
            
#        for output_key in self.output_target_mapper.keys():
#            labs=labels[self.output_target_mapper[output_key]]
#            preds=out[output_key]
#            preds = tf.expand_dims(preds,1)
#            batch_size = tf.shape(preds)[0]
#
#            if self.output_target_mapper[output_key] == 'p2':
#                lab_p2 = labs
#                pred_p2 = preds
#            elif self.output_target_mapper[output_key] == 'coordinates':
#                lab_co = labs
#                pred_co = preds
#
#        loss = (tf.map_fn(
#                lambda i: tf.reduce_min(tf.squeeze(self.custom_loss_elementwise(labels['nsources'][i],lab_co[i],pred_co[i])) +
#                tf.squeeze(self.custom_loss_elementwise(labels['nsources'][i],lab_p2[i],pred_p2[i]))),
#                tf.range(0,batch_size,1),dtype=tf.float32))
#        final_loss = tf.reduce_mean(loss,0)

        if self.weight_decay:
          # Add weight decay to the loss.
          l2_loss = self.weight_decay * tf.add_n(
              # loss is computed using fp32 for numerical stability.
              [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
          tf.summary.scalar('l2_loss', l2_loss)
          final_loss += l2_loss

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            
            optimizer = self.optimizer
            
            if self.loss_scale != 1:
              # When computing fp16 gradients, often intermediate tensor values are
              # so small, they underflow to 0. To avoid this, we multiply the loss by
              # loss_scale to make these tensor values loss_scale times bigger.
              scaled_grad_vars = optimizer.compute_gradients(final_loss * self.loss_scale)
        
              # Once the gradient computation is complete we can scale the gradients
              # back to the correct scale before passing them to the optimizer.
              unscaled_grad_vars = [(grad / self.loss_scale, var)
                                    for grad, var in scaled_grad_vars]
              minimize_op = optimizer.apply_gradients(unscaled_grad_vars, tf.train.get_global_step())
            else:
              minimize_op = optimizer.minimize(loss=final_loss,global_step=tf.train.get_global_step())

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group(minimize_op, update_ops)
            
            return tf.estimator.EstimatorSpec(mode=mode, loss=final_loss, train_op=train_op,
                                             eval_metric_ops=eval_metric_ops)
    
        # Add evaluation metrics (for EVAL mode)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=final_loss, eval_metric_ops=eval_metric_ops)      


class TimeHistory(tf.estimator.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_time_start = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)

def serving_input_fn():
    x = tf.placeholder(dtype=tf.float32, shape=[1,51,51,1], name='x')
    inputs = {'features': x }
    return tf.estimator.export.ServingInputReceiver(inputs,inputs)
