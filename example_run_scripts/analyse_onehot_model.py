from __future__ import absolute_import, division, print_function

import os

from NNFlow import OneHotModelAnalyser
#----------------------------------------------------------------------------------------------------


workdir_base    = 
name_subdir     = 'onehot_training'
file_name_model = 'onehot_training.ckpt'


path_to_data = os.path.join(workdir_base, name_subdir, 'training_data/val.hdf')


#----------------------------------------------------------------------------------------------------


save_dir = os.path.join(workdir_base, name_subdir, 'model/model_properties')


path_to_model = os.path.join(workdir_base, name_subdir, 'model', file_name_model)


#----------------------------------------------------------------------------------------------------


model_analyser = OneHotModelAnalyser(path_to_model)


model_analyser.save_variable_ranking(save_dir)
model_analyser.plot_heatmap(save_dir, 'heatmap', path_to_data)
