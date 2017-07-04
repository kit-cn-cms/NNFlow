# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================
 
from __future__ import absolute_import, division, print_function

import os

from NNFlow import OneHotModelAnalyser
#----------------------------------------------------------------------------------------------------


workdir_base    =
name_subdir     =
file_name_model =


number_of_output_neurons =


path_to_data =


#----------------------------------------------------------------------------------------------------


save_dir = os.path.join(workdir_base, name_subdir, 'model_properties')


path_to_model = os.path.join(workdir_base, name_subdir, 'model', file_name_model)
path_to_variablelist = os.path.join(workdir_base, name_subdir, 'training_data/variables.txt')
path_to_process_labels = os.path.join(workdir_base, name_subdir, 'training_data/process_labels.txt')


model_analyser = OneHotModelAnalyser(path_to_model, number_of_output_neurons)


model_analyser.save_variable_ranking(save_dir, path_to_variablelist)
model_analyser.plot_heatmap(save_dir, 'heatmap', path_to_data, path_to_process_labels)
