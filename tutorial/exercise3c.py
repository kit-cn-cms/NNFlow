import os

import numpy as np
from NNFlow.binary_mlp.binary_mlp import BinaryMLP
from NNFlow.binary_mlp.data_frame import DataFrame
from sklearn.metrics import roc_auc_score

## Begin Options which can be changed

# Define structure of the neural network
neurons_per_hidden_layer = 100
hidden_layers = 1
hlayers = [neurons_per_hidden_layer for _ in range(hidden_layers)]



# Available activation functions: 'elu', 'relu', 'tanh', 'sigmoid'
activation = 'relu'

# Maximum number of epochs, NN is trained
epochs = 1000

# If validation ROC score does not change for X epochs, stop training
early_stop = 20


# Parameter for dropout
keep_prob = 0.5


# Parameter for L2 regularization
beta = 1e-10

# Chosen optimizer
optimizer='adam'

# Chosen learning rate
lr=1e-3

# Chosen batch_size
batch_size=128


## End Options which can be changed


#----------------------------------------------------------------------------------------------------
# Program code

# GPU usage requirements
gpu_usage = dict()
gpu_usage['shared_machine']                           = True
gpu_usage['restrict_visible_devices']                 = False
gpu_usage['CUDA_VISIBLE_DEVICES']                     = '0'
gpu_usage['allow_growth']                             = True
gpu_usage['restrict_per_process_gpu_memory_fraction'] = False
gpu_usage['per_process_gpu_memory_fraction'] = 0.1

# Directory to save the results
modeldir = os.getcwd() + '/Aufgabe3c'
if not os.path.isdir(modeldir):
    if os.path.isdir(os.path.dirname(modeldir)):
        os.mkdir(modeldir)

# Selection of data sets
train_path = '/local/scratch/NNWorkshop/datasets/data_set_2/64/train.npy'
val_path = '/local/scratch/NNWorkshop/datasets/data_set_2/64/val.npy'

# Load data
train = DataFrame(np.load(train_path))
val = DataFrame(np.load(val_path))

# List containing variables 
variableList_path = '/local/scratch/NNWorkshop/datasets/data_set_2/64/vars.txt'
with open(variableList_path, 'r') as variableList_file:
  variableList = [variable.rstrip() for variable in variableList_file.readlines()]


# Define dictionaries
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


nn = BinaryMLP(**init_dict) 
nn.train(**train_dict)

nn.analyse_weights(variableList)