from __future__ import absolute_import, division, print_function

import os

from NNFlow import BinaryModelAnalyser
#----------------------------------------------------------------------------------------------------


workdir_base    = 
name_subdir     = 'binary_training'
file_name_model = 'binary_training.ckpt'


number_of_output_neurons = 1


path_to_data = os.path.join(workdir_base, name_subdir, 'training_data/val.npy')


#----------------------------------------------------------------------------------------------------


save_dir = os.path.join(workdir_base, name_subdir, 'model/model_properties')


path_to_model = os.path.join(workdir_base, name_subdir, 'model', file_name_model)
path_to_variablelist = os.path.join(workdir_base, name_subdir, 'training_data/variables.txt')


#----------------------------------------------------------------------------------------------------


model_analyser = BinaryModelAnalyser(path_to_model, number_of_output_neurons)


model_analyser.save_variable_ranking(save_dir, path_to_variablelist)
model_analyser.plot_output_distribution(path_to_data, save_dir, 'output_distribution')
