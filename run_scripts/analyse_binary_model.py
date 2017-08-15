# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================
 
from __future__ import absolute_import, division, print_function

import os
import sys

from NNFlow import BinaryModelAnalyser
#----------------------------------------------------------------------------------------------------


workdir_base    =
name_subdir     =
file_name_model =


#----------------------------------------------------------------------------------------------------


path_to_training_data_set   = os.path.join(workdir_base, name_subdir, 'data_sets/training_data_set.hdf')
path_to_validation_data_set = os.path.join(workdir_base, name_subdir, 'data_sets/validation_data_set.hdf')


save_dir = os.path.join(workdir_base, name_subdir, sys.argv[1], 'model_properties')


path_to_model = os.path.join(workdir_base, name_subdir, sys.argv[1], file_name_model)


#----------------------------------------------------------------------------------------------------
model_analyser = BinaryModelAnalyser(path_to_model)


model_analyser.save_variable_ranking(save_dir)
model_analyser.save_input_variables(save_dir)
model_analyser.save_unit_test_data(path_to_validation_data_set, save_dir)

model_analyser.plot_output_distribution(path_to_validation_data_set, save_dir, 'validation_output_distribution')
model_analyser.plot_training_validation_output_distribution(path_to_training_data_set, path_to_validation_data_set, save_dir, 'training_validation_output_distribution')
