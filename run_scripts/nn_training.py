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



#----------------------------------------------------------------------------------------------------
sys.path.append(NNFlow_base)
from binary_mlp.binary_mlp import BinaryMLP
from binary_mlp.data_frame import DataFrame
#----------------------------------------------------------------------------------------------------


n_hidden_layers =
n_neuron_per_layer =
hlayers = [n_neuron_per_layer for i in range(n_layers)]


### Available activation functions: 'elu', 'relu', 'tanh', 'sigmoid'
activation =


epochs =


early_stop =


### Parameter for dropout
keep_prob =


### Parameter for L2 regularization
beta =


#----------------------------------------------------------------------------------------------------


shared_machine = True

gpu_usage = dict()
gpu_usage['restrict_visible_devices']                 = False
gpu_usage['CUDA_VISIBLE_DEVICES']                     = '0'
gpu_usage['allow_growth']                             = True
gpu_usage['restrict_per_process_gpu_memory_fraction'] = False
gpu_usage['per_process_gpu_memory_fraction']          = 0.1


#----------------------------------------------------------------------------------------------------
modeldir             = os.path.join(workdir_base, name_subdir, 'model')
train                = os.path.join(workdir_base, name_subdir, 'training_data/train.npy')
val                  = os.path.join(workdir_base, name_subdir, 'training_data/val.npy')
path_to_variablelist = os.path.join(workdir_base, name_subdir, 'training_data/variables.txt')


optimizer='adam'
lr=1e-3
batch_size=128
#momentum=
#lr_decay=


#----------------------------------------------------------------------------------------------------
train = DataFrame(np.load(train))
val = DataFrame(np.load(val))
test = DataFrame(np.load(test))

with open(path_to_variablelist, 'r') as file_variablelist:
    variablelist = [variable.rstrip() for variable in file_variablelist.readlines()]


#----------------------------------------------------------------------------------------------------
init_dict = {'n_variables' : train.nvariables,
             'h_layers'    : hlayers,
             'savedir'     : modeldir,
             'activation'  : activation
             }


train_dict = {'train_data' : train,
              'val_data'   : val,
              'epochs'     : epochs,
              'batch_size' : batch_size,
              'lr'         : lr,
              'optimizer'  : optimizer,
              'early_stop' : early_stop,
              'keep_prob'  : keep_prob,
              'beta'       : beta
              }


if shared_machine:
    train_dict['gpu_usage'] = gpu_usage

if optimizer=='momentum':
    train_dict['momentum'] = momentum

if optimizer=='momentum' or optimizer=='gradientdescent':
    train_dict['lr_decay'] = lr_decay


nn = BinaryMLP(**init_dict) 
nn.train(**train_dict)

nn.analyse_weights(variablelist)
