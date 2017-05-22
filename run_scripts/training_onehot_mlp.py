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
from onehot_mlp.onehot_mlp import OneHotMLP
from onehot_mlp.data_frame import DataFrame
#----------------------------------------------------------------------------------------------------


optname = 'Momentum'
optimizer_options = []
act_func = 'elu'
N_EPOCHS = 1000
batch_size = 1000
learning_rate = 5e-3
keep_prob = 0.7
beta = 1e-8
enable_early='yes'
early_stop = 15
decay_learning_rate = 'no'
lrate_decay_options = []
batch_decay = 'no'
batch_decay_options = []

hidden_layers = [200, 200, 200, 200]


#----------------------------------------------------------------------------------------------------


gpu_usage = dict()

gpu_usage['shared_machine']                           = True

gpu_usage['restrict_visible_devices']                 = False
gpu_usage['CUDA_VISIBLE_DEVICES']                     = '0' 
gpu_usage['allow_growth']                             = True
gpu_usage['restrict_per_process_gpu_memory_fraction'] = False
gpu_usage['per_process_gpu_memory_fraction']          = 0.1


#----------------------------------------------------------------------------------------------------
model_location = os.path.join(workdir_base, name_subdir, 'model')
train_path     = os.path.join(workdir_base, name_subdir, 'training_data/train.npy')
val_path       = os.path.join(workdir_base, name_subdir, 'training_data/val.npy')
branchlist     = os.path.join(workdir_base, name_subdir, 'training_data/variables.txt')
labels_path    = os.path.join(workdir_base, name_subdir, 'training_data/process_labels.txt')


with open(labels_path, 'r') as file_labels:
    labels = [label.rstrip() for label in file_labels.readlines()]

outsize = len(labels)

sig_weight = np.float32(1) #TODO
bg_weight = np.float32(1)  #TODO

train = DataFrame(np.load(train_path), out_size=outsize)
val   = DataFrame(np.load(val_path), out_size=outsize)


#----------------------------------------------------------------------------------------------------


cl = OneHotMLP(train.nfeatures, hidden_layers, outsize, model_location, 
        labels_text=labels, branchlist=branchlist, sig_weight=sig_weight,
        bg_weight=bg_weight, act_func=act_func)
cl.train(train, val, optimizer=optname, epochs=N_EPOCHS, batch_size=batch_size, 
        learning_rate=learning_rate, keep_prob=keep_prob, beta=beta, 
        out_size=outsize, optimizer_options=optimizer_options,
        enable_early=enable_early, early_stop=early_stop,
        decay_learning_rate=decay_learning_rate,
        dlrate_options=lrate_decay_options, batch_decay=batch_decay,
        batch_decay_options=batch_decay_options, gpu_usage=gpu_usage)


# implemented Optimizers: 
# 'Adam':               Adam Optimizer
# 'GradDescent':        Gradient Descent Optimizer
# 'Adagrad':            Adagrad Optimizer
# 'Adadelta':           Adadelta Optimizer
# 'Momentum':           Momentum Optimizer
# Optimizer options may have different data types for different optimizers.
# 'Adam':               [beta1=0.9 (float), beta2=0.999 (float), epsilon=1e-8 (float)]
# 'GradDescent':        []
# 'Adagrad':            [initial_accumulator_value=0.1 (float)]
# 'Adadelta':           [rho=0.95 (float), epsilon=1e-8 (float)]
# 'Momentum':           [momentum=0.9 (float), use_nesterov=False (bool)]
# For information about these parameters please refer to the TensorFlow
# documentation.
# Choose normalization from 'minmax' or 'gaussian'.
