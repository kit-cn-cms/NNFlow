# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================

from __future__ import absolute_import, division, print_function

import os

import NNFlow
from NNFlow.ttH_ttbb_definitions import definitions
#----------------------------------------------------------------------------------------------------


### Path to the directory where your project is located.
workdir_base =

### Each neural network configuration gets a separate directory.
### Specify the name for the subdirectory of this neural network configuration here.
name_subdir  =


### You can choose from categories defined in ttH_ttbb_definitions.definitions.py
### To keep all events, specify "jet_btag_category = 'all'"
jet_btag_category =


### If you don't want to keep all processes, specify the processes you want to keep as a list.
### To keep all processes, specify "selected_processes = 'all'"
selected_processes =


### Binary or multiclass classification (binary_classification = True/False).
### If you want to do a binary classification, specify the process that should be treated as signal.
binary_classification        =
#binary_classification_signal =


# If desired, only a subset of the available variables is saved in the data set for the training.
# Option 0: All variables are kept (select_variables = False).
# Option 1: Provide a list with variables you want to keep as a text file (select_variables = 'include').
# Option 2: Provide a list with variables you want to drop as a text file (select_variables = 'exclude').
select_variables =
#path_to_variable_list =


### If you want to select events with cuts on certain variables, specify "cutbased_event_selection = True".
### The condition has to be given as a string (e.g. for cuts on the variables A and B: 'A > 5 and B <= 3').
cutbased_event_selection =
#cutbased_event_selection_condition =


### If you want to merge several processes for the training, you can create new process labels for the merged processes here (specify "create_new_process_labels = True" in this case).
### The variable "new_process_labels" has to be a dictionary, e.g. new_process_labels = {'ttHF':['ttbb', 'tt2b', 'ttb']}.
create_new_process_labels =
#new_process_labels =


### Specify if you want to drop events with negative weights for the training.
### Note: If you don't drop events with negative weights, the training may become unstable.
drop_events_negative_weights = True


#----------------------------------------------------------------------------------------------------


save_path = os.path.join(workdir_base, name_subdir, 'data_sets')


path_to_merged_data_set = os.path.join(workdir_base, 'HDF5_files/data_set_merged.hdf')


### Specify the MC weights that will be applied in the training.
### The weights you want to use here have to be included in the "weights_to_keep" list in preprocessing_1_root_to_HDF5.py
### The weights will be normalized so that the sum of weights is equal for each process.
weights_to_be_applied = definitions.default_weight_list()


#----------------------------------------------------------------------------------------------------
if not os.path.isdir(save_path):
    if os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(save_path)
#----------------------------------------------------------------------------------------------------
function_call_dict = {'jet_btag_category'            : jet_btag_category,
                      'selected_processes'           : selected_processes,
                      'binary_classification'        : binary_classification,
                      'select_variables'             : select_variables,
                      'cutbased_event_selection'     : cutbased_event_selection,
                      'create_new_process_labels'    : create_new_process_labels,
                      'save_path'                    : save_path,
                      'path_to_merged_data_set'      : path_to_merged_data_set,
                      'weights_to_be_applied'        : weights_to_be_applied,
                      'drop_events_negative_weights' : drop_events_negative_weights
                      }


if binary_classification:
    function_call_dict['binary_classification_signal'] = binary_classification_signal

if select_variables != False:
    function_call_dict['path_to_variable_list'] = path_to_variable_list

if cutbased_event_selection:
    function_call_dict['cutbased_event_selection_condition'] = cutbased_event_selection_condition

if create_new_process_labels:
    function_call_dict['new_process_labels'] = new_process_labels


NNFlow.preprocessing.create_data_set_for_training(**function_call_dict)
