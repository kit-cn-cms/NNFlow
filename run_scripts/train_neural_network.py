# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================
 
from __future__ import absolute_import, division, print_function

import os
import sys

import NNFlow
#----------------------------------------------------------------------------------------------------


workdir_base =
name_subdir  =


### Network type: 'binary' or 'one-hot'
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


save_path  = os.path.join(workdir_base, name_subdir, 'model')
model_name = name_subdir


path_to_training_data_set   = os.path.join(workdir_base, name_subdir, 'training_data/train.npy')
path_to_validation_data_set = os.path.join(workdir_base, name_subdir, 'training_data/val.npy')


if network_type == 'one-hot':
    path_to_process_names = os.path.join(workdir_base, name_subdir, 'training_data/process_labels.txt')


#----------------------------------------------------------------------------------------------------
if network_type == 'binary':
    number_of_output_neurons = 1

elif network_type == 'one-hot':
    with open(path_to_process_names, 'r') as file_process_names:
        process_names = file_process_names.readlines()
    number_of_output_neurons = len(process_names)
#----------------------------------------------------------------------------------------------------
if not os.path.isdir(save_path):
    if os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(save_path)
#----------------------------------------------------------------------------------------------------
train_dict = {'save_path'                   : save_path,
              'model_name'                  : model_name,
              'network_type'                : network_type,
              'number_of_output_neurons'    : number_of_output_neurons,
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


train_neural_network(**train_dict)
