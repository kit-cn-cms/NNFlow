# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================

from __future__ import absolute_import, division, print_function

import os
import sys
#----------------------------------------------------------------------------------------------------


NNFlow_base  =
workdir_base =


#----------------------------------------------------------------------------------------------------
sys.path.append(NNFlow_base)
from preprocessing.preprocessing import create_data_set_for_training
from definitions import definitions
#----------------------------------------------------------------------------------------------------



### The input data sets have to be provided as a dictionary in the following format:
### input_datasets = {'process_name':'filename'}
input_data_sets =


### You can choose from categories defined in definitions.definitions.jet_btag_category
### To keep all events, specify "jet_btag_category = 'all'"
jet_btag_category =


### If you don't want to keep all processes, specify the processes you want to keep.
### To keep all processes, specify "selected_processes = 'all'"
selected_processes =


### Binary or multinomial classification (binary_classification = True/False).
### If you want to do a binary classification, specify the process which should be treated as signal.
binary_classification        =
#binary_classification_signal =


# If desired, only a subset of the available variables is saved in the data set for the training.
# Option 0: All variables are kept (select_variables = False).
# Option 1: Provide a list with variables you want to keep (select_variables = 'include').
# Option 2: Provide a list with variables you want to drop (select_variables = 'exclude').
select_variables =
#path_to_variable_list =


#----------------------------------------------------------------------------------------------------
save_path = os.path.join(workdir_base, 'training_data')


path_to_inputfiles = os.path.join(workdir_base, 'HDF5_files')
path_to_merged_data_set = os.path.join(workdir_base, 'HDF5_files/data_set_merged.hdf')


path_to_weight_variables = os.path.join(NNFlow_base, 'definitions/excluded_variables/weight_variables.txt')
weights_to_be_applied = definitions.default_weight_list()


#----------------------------------------------------------------------------------------------------
function_call_dict = {'input_data_sets'              : input_data_sets,
                      'jet_btag_category'            : jet_btag_category,
                      'selected_processes'           : selected_processes,
                      'binary_classification'        : binary_classification,
                      'select_variables'             : select_variables,
                      'save_path'                    : save_path,
                      'path_to_inputfiles'           : path_to_inputfiles,
                      'path_to_merged_data_set'      : path_to_merged_data_set,
                      'path_to_weight_variables'     : path_to_weight_variables,
                      'weights_to_be_applied'        : weights_to_be_applied,
                      }


if binary_classification:
    function_call_dict['binary_classification_signal'] = binary_classification_signal

if select_variables != False:
    function_call_dict['path_to_variable_list'] = path_to_variable_list


create_data_set_for_training(**function_call_dict)
