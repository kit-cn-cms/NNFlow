import os

import numpy as np
from NNFlow.onehot_mlp.onehot_mlp import OneHotMLP
from NNFlow.onehot_mlp.data_frame import DataFrame


## Begin Options which can be changed

# Define structure of the neural network
#neurons_per_hidden_layer = 100
#hidden_layers = 1
#hlayers = [neurons_per_hidden_layer for _ in range(hidden_layers)]
hlayers = [200, 200, 200, 200]


# Available activation functions: 'elu', 'relu', 'tanh', 'sigmoid'
activation = 'elu'

# Optimizer and its optimizer options
# Chosen optimizer
optimizer='Momentum'
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
optimizer_options = []


# Maximum number of epochs, NN is trained
epochs = 1000

# If validation ROC score does not change for X epochs, stop training
enable_early='yes'
early_stop = 15


# Parameter for dropout
keep_prob = 0.7

# Parameter for L2 regularization
beta = 1e-8

# Chosen learning rate
lr=5
# Reduce learning rate
decay_lr = 'no'
lrate_decay_options = []

# Chosen batch_size
batch_size=1000
# Reduce batch_size
batch_decay = 'no'
batch_decay_options = []


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
modeldir = os.getcwd() + '/Higgschallenge'
if not os.path.isdir(modeldir):
    if os.path.isdir(os.path.dirname(modeldir)):
        os.mkdir(modeldir)

# List containing sample categories
labelList_path = '/local/scratch/NNWorkshop/datasets/higgschallenge/process_labels.txt'
with open(labelList_path, 'r') as labelList_file:
  labelList = [label.rstrip() for label in labelList_file.readlines()]

# Set length of categories
outsize = len(labelList)


# Selection of data sets
train_path = '/local/scratch/NNWorkshop/datasets/higgschallenge/train.npy'
val_path = '/local/scratch/NNWorkshop/datasets/higgschallenge/val.npy'

# Load data
train = DataFrame(np.load(train_path), out_size=outsize)
val = DataFrame(np.load(val_path), out_size=outsize)

# List containing variables 
variableList_path = '/local/scratch/NNWorkshop/datasets/higgschallenge/variables.txt'
#with open(variableList_path, 'r') as variableList_file:
#  variableList = [variable.rstrip() for variable in variableList_file.readlines()]
  

cl = OneHotMLP(train.nfeatures, hlayers, outsize, modeldir, 
        labels_text=labelList, branchlist=variableList_path,
        act_func=activation)


cl.train(train, val, optimizer=optimizer, epochs=epochs, batch_size=batch_size, 
        learning_rate=lr, keep_prob=keep_prob, beta=beta, 
        out_size=outsize, optimizer_options=optimizer_options,
        enable_early=enable_early, early_stop=early_stop,
        decay_learning_rate=decay_lr,
        dlrate_options=lrate_decay_options, batch_decay=batch_decay,
batch_decay_options=batch_decay_options, gpu_usage=gpu_usage)







