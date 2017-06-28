# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================
 
from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
#----------------------------------------------------------------------------------------------------


NNFlow_base  =
workdir_base =
name_subdir  =


#----------------------------------------------------------------------------------------------------
sys.path.append(NNFlow_base)
from mlp.mlp import MLP
#----------------------------------------------------------------------------------------------------


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


#----------------------------------------------------------------------------------------------------
### The following optimizers are avialable.
### Depending on the chosen optimizer, you have to provide the additional options.
### If you do not want to use your own values, you can use the values that are suggested here.
### 'Adam'               optimizer_options['beta1']=0.9, optimizer_options['beta2']=0.999, optimizer_options['epsilon']=1e-8
### 'Adadelta'           optimizer_options['rho']=0.95, optimizer_options['epsilon']=1e-8
### 'Adagrad'            optimizer_options['initial_accumulator_value']=0.1
### 'GradientDescent'
### 'Momentum'           optimizer_options['momentum']=0.9, optimizer_options['use_nesterov']=False


optimizer_options = dict()

optimizer_options['optimizer_name'] =


optimizer_options['learning_rate_decay'] =

### The following options have to be provided if you want to use learning rate decay.
#optimizer_options['learning_rate_decay_initial_value'] =
#optimizer_options['learning_rate_decay_rate']          =
#optimizer_options['learning_rate_decay_steps']         =

### The following value has to be provided if you do not want to use learning rate decay.
optimizer_options['learning_rate'] =


#----------------------------------------------------------------------------------------------------


gpu_usage = dict()

gpu_usage['shared_machine']                           = True

gpu_usage['restrict_visible_devices']                 = False
gpu_usage['CUDA_VISIBLE_DEVICES']                     = '0'
gpu_usage['allow_growth']                             = True
gpu_usage['restrict_per_process_gpu_memory_fraction'] = False
gpu_usage['per_process_gpu_memory_fraction']          = 0.1


#----------------------------------------------------------------------------------------------------


save_path  = os.path.join(workdir_base, name_subdir, 'model')
model_name = name_subdir


batch_size_classification = 1000


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
if not os.path.isdir(modeldir):
    if os.path.isdir(os.path.dirname(modeldir)):
        os.mkdir(modeldir)
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
              'optimizer_options'           : optimizer_options,
              'batch_size_training'         : batch_size_training,
              'batch_size_classification'   : batch_size_classification,
              'gpu_usage'                   : gpu_usage
              }


nn = MLP() 
nn.train(**train_dict)
