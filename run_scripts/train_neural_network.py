# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================
 
from __future__ import absolute_import, division, print_function

import os
import sys
import datetime

import NNFlow
#----------------------------------------------------------------------------------------------------


workdir_base =
name_subdir  =


### Network type: 'binary' or 'multiclass'
network_type =


number_of_hidden_layers     =
number_of_neurons_per_layer =
hidden_layers               = [number_of_neurons_per_layer for i in range(number_of_hidden_layers)]


### Available activation functions: 'elu', 'relu', 'tanh', 'sigmoid', 'softplus'
activation_function_name =


early_stopping_intervall =


### Parameter for dropout
dropout_keep_probability =


### Parameter for L2 regularization
l2_regularization_beta =


batch_size_training =


optimizer =


#----------------------------------------------------------------------------------------------------


save_path  = os.path.join(workdir_base, name_subdir, 'model_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
model_name = name_subdir


path_to_training_data_set   = os.path.join(workdir_base, name_subdir, 'data_sets/training_data_set.hdf')
path_to_validation_data_set = os.path.join(workdir_base, name_subdir, 'data_sets/validation_data_set.hdf')


#----------------------------------------------------------------------------------------------------
if not os.path.isdir(save_path):
    if os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(save_path)
#----------------------------------------------------------------------------------------------------
train_dict = {'save_path'                   : save_path,
              'model_name'                  : model_name,
              'network_type'                : network_type,
              'hidden_layers'               : hidden_layers,
              'activation_function_name'    : activation_function_name,
              'dropout_keep_probability'    : dropout_keep_probability,
              'l2_regularization_beta'      : l2_regularization_beta,
              'early_stopping_intervall'    : early_stopping_intervall,
              'path_to_training_data_set'   : path_to_training_data_set,
              'path_to_validation_data_set' : path_to_validation_data_set,
              'optimizer'                   : optimizer,
              'batch_size_training'         : batch_size_training,
              }


NNFlow.train_neural_network(**train_dict)
