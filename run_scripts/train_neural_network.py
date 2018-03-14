# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================
 
from __future__ import absolute_import, division, print_function

import os
import datetime

import NNFlow
#----------------------------------------------------------------------------------------------------


### Path to the directory where your project is located and name of the subdirectory of this neural network configuration.
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


### The optimizer has to be an instance of a NNFlow optimizer.
### The NNFlow optimizers are wrappers around TensorFlow optimizers.
### The following optimizers are available:
### NNFlow.optimizers.AdadeltaOptimizer(learning_rate, rho = 0.95, epsilon = 1e-08)
### NNFlow.optimizers.AdagradOptimizer(learning_rate, initial_accumulator_value = 0.1)
### NNFlow.optimizers.AdamOptimizer(learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08)
### NNFlow.optimizers.GradientDescentOptimizer(learning_rate)
### NNFlow.optimizers.MomentumOptimizer(learning_rate, momentum, use_nesterov = False)
optimizer =


#----------------------------------------------------------------------------------------------------


model_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


save_path  = os.path.join(workdir_base, name_subdir, 'model_' + model_id)
model_name = name_subdir + '_' + model_id


path_to_training_data_set   = os.path.join(workdir_base, name_subdir, 'data_sets/training_data_set.hdf')
path_to_validation_data_set = os.path.join(workdir_base, name_subdir, 'data_sets/validation_data_set.hdf')


### The batch size for evaluating the neural network can be chosen independently from the batch size for the training.
### It can be chosen as high as the available memory allows.
### The default value is 200000.
#batch_size_classification =


### You can limit the resources used for training with a session config.
### If you don't specify a session config, the following settings will be used: visible_devices = 'all', allow_growth = True, per_process_gpu_memory_fraction = None
#session_config = NNFlow.session_config.SessionConfig(visible_devices = 'all', allow_growth = True, per_process_gpu_memory_fraction = None)


#----------------------------------------------------------------------------------------------------
if not os.path.isdir(save_path):
    if os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(save_path)
#----------------------------------------------------------------------------------------------------
train_dict = {'save_path'                   : save_path,
              'model_name'                  : model_name,
              'model_id'                    : model_id,
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
#              'batch_size_classification'   : batch_size_classification,
#              'session_config'              : session_config,
              }


NNFlow.train_neural_network(**train_dict)
