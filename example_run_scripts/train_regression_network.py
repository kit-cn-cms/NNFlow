# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================

import os
import datetime

import NNFlow

# ----------------------------------------------------------------------------------------------------


workdir_base = '/storage/9/jschindler/'
name_subdir = 'NN_v1'


number_of_hidden_layers = 3
number_of_neurons_per_layer = 200
hidden_layers = [number_of_neurons_per_layer for i in range(number_of_hidden_layers)]

### Available activation functions: 'elu', 'relu', 'tanh', 'sigmoid', 'softplus'
activation_function_name = 'elu'

early_stopping_intervall = 10

### Parameter for dropout
dropout_keep_probability = 1.

### Parameter for L2 regularization
l2_regularization_beta = 0

batch_size_training = 500
optimizer = None

parameter = 'Evt_blr_ETH'
# ----------------------------------------------------------------------------------------------------


model_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

save_path = os.path.join(workdir_base, name_subdir, 'tth/model_' + model_id)
model_name = name_subdir + '_' + model_id

path_to_training_data_set = os.path.join(workdir_base, 'data_sets_2/training_data_set.hdf')
path_to_validation_data_set = os.path.join(workdir_base, 'data_sets_2/validation_data_set.hdf')
path_to_test_data_set =  os.path.join(workdir_base, 'data_sets_2/test_data_set.hdf')
# ----------------------------------------------------------------------------------------------------
if not os.path.isdir(save_path):
    if os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(save_path)
# ----------------------------------------------------------------------------------------------------
train_dict = {'save_path': save_path,
              'model_id': model_id,
              'hidden_layers': hidden_layers,
              'activation_function_name': activation_function_name,
              'dropout_keep_probability': dropout_keep_probability,
              'l2_regularization_beta': l2_regularization_beta,
              'early_stopping_intervall': early_stopping_intervall,
              'path_to_training_data_set': path_to_training_data_set,
              'path_to_validation_data_set': path_to_validation_data_set,
              'path_to_test_data_set' : path_to_test_data_set,
              'parameter': parameter,
              'batch_size_training': batch_size_training,
              }

NNFlow.train_regression_network(**train_dict)