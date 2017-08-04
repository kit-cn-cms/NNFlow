# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================
 
from __future__ import absolute_import, division, print_function

import os

from NNFlow import MulticlassModelAnalyser
#----------------------------------------------------------------------------------------------------


workdir_base    =
name_subdir     =
file_name_model =


#----------------------------------------------------------------------------------------------------


path_to_validation_data_set = os.path.join(workdir_base, name_subdir, 'data_sets/validation_data_set.hdf')


save_dir = os.path.join(workdir_base, name_subdir, 'model/model_properties')


path_to_model = os.path.join(workdir_base, name_subdir, 'model', file_name_model)


#----------------------------------------------------------------------------------------------------
model_analyser = MulticlassModelAnalyser(path_to_model)


model_analyser.save_variable_ranking(save_dir)
model_analyser.save_input_variables(save_dir)
model_analyser.save_output_labels(save_dir)
model_analyser.plot_confusion_matrix(save_dir, 'confusion_matrix', path_to_validation_data_set)
