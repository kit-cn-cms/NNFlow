from __future__ import absolute_import, division, print_function

import os
import sys

import NNFlow
#----------------------------------------------------------------------------------------------------


workdir_base =
name_subdir  = 'binary_training'


### Network type: 'binary' or 'one-hot'
network_type = 'binary'


number_of_hidden_layers     = 1
number_of_neurons_per_layer = 250
hidden_layers               = [number_of_neurons_per_layer for i in range(number_of_hidden_layers)]


### Available activation functions: 'elu', 'relu', 'tanh', 'sigmoid', 'softplus'
activation_function_name = 'elu'


early_stopping_intervall = 20


### Parameter for dropout
dropout_keep_probability = 0.7


### Parameter for L2 regularization
l2_regularization_beta = 3e-7


batch_size_training = 125


optimizer = NNFlow.optimizers.AdamOptimizer(learning_rate=1e-3)


#----------------------------------------------------------------------------------------------------


save_path  = os.path.join(workdir_base, name_subdir, 'model')
model_name = name_subdir


path_to_training_data_set   = os.path.join(workdir_base, name_subdir, 'training_data/train.hdf')
path_to_validation_data_set = os.path.join(workdir_base, name_subdir, 'training_data/val.hdf')


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
