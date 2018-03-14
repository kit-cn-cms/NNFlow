# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================

from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict

import NNFlow
#----------------------------------------------------------------------------------------------------


### Path to the directory where your project is located.
workdir_base =


### The input data sets have to be provided as a dictionary in the following format:
### input_datasets = {'process_name':['filename1', 'filename2', ...]}
input_data_sets =


### If you want to select your training events with cuts on certrain variables, you have to specify the variables that will be used to make cuts here.
### The selection of events with the variables "N_Jets" and "N_BTagsM" is handled in a different way, these variables don't have to be added to this list.
cutbased_event_selection = False
#columns_for_cutbased_event_selection =


#----------------------------------------------------------------------------------------------------


path_to_inputfiles = os.path.join(workdir_base, 'HDF5_files')
path_to_merged_data_set = os.path.join(workdir_base, 'HDF5_files/data_set_merged.hdf')


#----------------------------------------------------------------------------------------------------
function_call_dict = {'input_data_sets'         : input_data_sets,
                      'path_to_inputfiles'      : path_to_inputfiles,
                      'path_to_merged_data_set' : path_to_merged_data_set
                      }

if cutbased_event_selection:
    function_call_dict['columns_for_cutbased_event_selection'] = columns_for_cutbased_event_selection


NNFlow.preprocessing.merge_data_sets(**function_call_dict)
