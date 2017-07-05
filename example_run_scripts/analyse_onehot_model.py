from __future__ import absolute_import, division, print_function

import os

from NNFlow import OneHotModelAnalyser
#----------------------------------------------------------------------------------------------------


workdir_base    = 
name_subdir     = 'onehot_training'
file_name_model = 'onehot_training.ckpt'


number_of_output_neurons = 6


path_to_data = os.path.join(workdir_base, name_subdir, 'training_data/val.npy')


#----------------------------------------------------------------------------------------------------


save_dir = os.path.join(workdir_base, name_subdir, 'model_properties')


path_to_model = os.path.join(workdir_base, name_subdir, 'model', file_name_model)
path_to_variablelist = os.path.join(workdir_base, name_subdir, 'training_data/variables.txt')
path_to_process_labels = os.path.join(workdir_base, name_subdir, 'training_data/process_labels.txt')


#----------------------------------------------------------------------------------------------------

with open(path_to_process_labels, 'r') as file_process_labels:
    process_labels = [process.rstrip() for process in file_process_labels.readlines()]

#----------------------------------------------------------------------------------------------------


if not os.path.isdir(save_dir):
    if os.path.isdir(os.path.dirname(save_dir)):
        os.mkdir(save_dir)

#----------------------------------------------------------------------------------------------------


model_analyser = OneHotModelAnalyser(path_to_model, number_of_output_neurons)


model_analyser.save_variable_ranking(save_dir, path_to_variablelist)
model_analyser.plot_heatmap(save_dir, 'heatmap', path_to_data, process_labels)
