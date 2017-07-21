from __future__ import absolute_import, division, print_function

import os

from NNFlow import MulticlassModelAnalyser
#----------------------------------------------------------------------------------------------------


workdir_base    = 
name_subdir     = 'multiclass_training'
file_name_model = 'multiclass_training.ckpt'


path_to_data = os.path.join(workdir_base, name_subdir, 'data_sets/validation_data_set.hdf')


#----------------------------------------------------------------------------------------------------


save_dir = os.path.join(workdir_base, name_subdir, 'model/model_properties')


path_to_model = os.path.join(workdir_base, name_subdir, 'model', file_name_model)


#----------------------------------------------------------------------------------------------------


model_analyser = MulticlassModelAnalyser(path_to_model)


model_analyser.save_variable_ranking(save_dir)
model_analyser.plot_confusion_matrix(save_dir, 'confusion_matrix', path_to_data)
