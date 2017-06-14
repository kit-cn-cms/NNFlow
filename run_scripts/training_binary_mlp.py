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
from mlp.binary_mlp import BinaryMLP
from mlp.data_frame_binary_mlp import DataFrame
#----------------------------------------------------------------------------------------------------


n_hidden_layers =
n_neuron_per_layer =
hlayers = [n_neuron_per_layer for i in range(n_hidden_layers)]


### Available activation functions: 'elu', 'relu', 'tanh', 'sigmoid'
activation =


epochs =


early_stop =


### Parameter for dropout
keep_prob =


### Parameter for L2 regularization
beta =


#----------------------------------------------------------------------------------------------------


gpu_usage = dict()

gpu_usage['shared_machine']                           = True

gpu_usage['restrict_visible_devices']                 = False
gpu_usage['CUDA_VISIBLE_DEVICES']                     = '0'
gpu_usage['allow_growth']                             = True
gpu_usage['restrict_per_process_gpu_memory_fraction'] = False
gpu_usage['per_process_gpu_memory_fraction']          = 0.1


#----------------------------------------------------------------------------------------------------


modeldir             = os.path.join(workdir_base, name_subdir, 'model')
train_path           = os.path.join(workdir_base, name_subdir, 'training_data/train.npy')
val_path             = os.path.join(workdir_base, name_subdir, 'training_data/val.npy')


optimizer='adam'
lr=1e-3
batch_size=128
#momentum=
#lr_decay=


#----------------------------------------------------------------------------------------------------
if not os.path.isdir(modeldir):
    if os.path.isdir(os.path.dirname(modeldir)):
        os.mkdir(modeldir)
#----------------------------------------------------------------------------------------------------
train = DataFrame(np.load(train_path))
val = DataFrame(np.load(val_path))
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
              'beta'       : beta,
              'gpu_usage'  : gpu_usage
              }


if optimizer=='momentum':
    train_dict['momentum'] = momentum

if optimizer=='momentum' or optimizer=='gradientdescent':
    train_dict['lr_decay'] = lr_decay


nn = BinaryMLP(**init_dict) 
nn.train(**train_dict)
