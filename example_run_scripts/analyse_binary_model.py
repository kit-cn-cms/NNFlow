from __future__ import absolute_import, division, print_function

import os

from NNFlow import BinaryModelAnalyser
#----------------------------------------------------------------------------------------------------


workdir_base    = 
name_subdir     = 'binary_training'
file_name_model = 'binary_training.ckpt'


#----------------------------------------------------------------------------------------------------


path_to_training_data_set   = os.path.join(workdir_base, name_subdir, 'training_data/train.npy')
path_to_validation_data_set = os.path.join(workdir_base, name_subdir, 'training_data/val.npy')


save_dir = os.path.join(workdir_base, name_subdir, 'model/model_properties')


path_to_model = os.path.join(workdir_base, name_subdir, 'model', file_name_model)


#----------------------------------------------------------------------------------------------------


model_analyser = BinaryModelAnalyser(path_to_model)


model_analyser.save_variable_ranking(save_dir)
model_analyser.plot_training_validation_output_distribution(path_to_training_data_set, path_to_validation_data_set, save_dir, 'training_validation_output_distribution')
