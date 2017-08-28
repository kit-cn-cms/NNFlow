# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================
 
from __future__ import absolute_import, division, print_function

import os
import sys

import NNFlow
#----------------------------------------------------------------------------------------------------


workdir_base =
name_subdir  =


#----------------------------------------------------------------------------------------------------


model_id        = sys.argv[1]
file_name_model = name_subdir + '_' + model_id + '.ckpt'


path_to_validation_data_set = os.path.join(workdir_base, name_subdir, 'data_sets/validation_data_set.hdf')


save_dir_model_properties = os.path.join(workdir_base, name_subdir, 'model_' + model_id, 'model_properties')
save_dir_plots            = os.path.join(workdir_base, name_subdir, 'model_' + model_id, 'plots')


path_to_model = os.path.join(workdir_base, name_subdir, 'model_' + model_id, file_name_model)


#----------------------------------------------------------------------------------------------------
model_analyser = NNFlow.MulticlassModelAnalyser(path_to_model)


model_analyser.save_variable_ranking(save_dir_model_properties)
model_analyser.save_unit_test_data(path_to_validation_data_set, save_dir_model_properties)

model_analyser.plot_confusion_matrix(save_dir_plots, 'confusion_matrix', path_to_validation_data_set)
