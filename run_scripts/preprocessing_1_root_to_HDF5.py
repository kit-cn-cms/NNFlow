# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================

from __future__ import absolute_import, division, print_function

import os

import NNFlow
from NNFlow.ttH_ttbb_definitions import definitions
from NNFlow.ttH_ttbb_definitions import excluded_variables
from NNFlow.ttH_ttbb_definitions import vector_variables
#----------------------------------------------------------------------------------------------------


### Specify the path to the directory where you want to locate your project.
workdir_base =


### Provide the file name for the output file WITHOUT file name extension.
filename_outputfile =


### Specify the path to the directory where the ROOT files are located.
path_to_inputfiles =


### The filenames of the input files have to be provided as a list of strings.
filenames_inputfiles =


### The tree names in the ROOT file have to be provided as a list of strings.
treenames =


### Here you can specify the conditions to split the data set into subprocesses.
### The conditions for splitting have to be provided as a dictionary (see example in ttH_ttbb_definitions.definitions.py).
split_data_set = False
#conditions_for_splitting = definitions.ttbar_processes()


### MEM
mem_database_path              = None
mem_database_sample_name       = None
mem_database_sample_index_file = None


#----------------------------------------------------------------------------------------------------


save_path = os.path.join(workdir_base, 'HDF5_files')


### You can't use generator level variables to train neural networks because they are not available in measurement data.
### The generator level variables given here will be removed from the data set.
list_of_generator_level_variables       = excluded_variables.generator_level_variables()


### If you also want to remove other variables from the data set for some reason, you can include them in this list.
list_of_other_always_excluded_variables = excluded_variables.other_always_excluded_variables()


### List of variables that are a vector in the ROOT file.
### For more details, see comments below.
list_of_vector_variables_lepton         = vector_variables.lepton_variables()
list_of_vector_variables_jet            = vector_variables.jet_variables()


### Variables with names beginning with "Weight" will be deleted by default.
### If you want to keep certain weights, you have to specify them in this list.
weights_to_keep = definitions.default_weight_list()


### Neural networks need a fixed number of input variables.
### If the number of jets or leptons varies, you can only include the properties of the objects that exist in all events.
### The vector variables have to be specified above.
### You can limit the number of saved objects here to save resources.
### You don't have to worry if you save more objects than available, the corresponding variables will be removed later.
number_of_saved_jets    = 6
number_of_saved_leptons = 1


### The data set will be split into three parts.
### The splitting into a training/validation and a test data set is done with the condition specified in NNFlow/ttH_ttbb_definitions/definitions.py
### The events in the training/validation part will be randomly distributed into the training and the validation data set.
### The percentage of the events in the training/validation part that goes into the validation data set can be specified here.
percentage_validation = 20


#----------------------------------------------------------------------------------------------------
if not os.path.isdir(save_path):
    if os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(save_path)
#----------------------------------------------------------------------------------------------------
function_call_dict = {'filename_outputfile'                     : filename_outputfile,
                      'path_to_inputfiles'                      : path_to_inputfiles,
                      'filenames_inputfiles'                    : filenames_inputfiles,
                      'treenames'                               : treenames,
                      'split_data_set'                          : split_data_set,
                      'save_path'                               : save_path,
                      'list_of_generator_level_variables'       : list_of_generator_level_variables,
                      'list_of_other_always_excluded_variables' : list_of_other_always_excluded_variables,
                      'list_of_vector_variables_lepton'         : list_of_vector_variables_lepton,
                      'list_of_vector_variables_jet'            : list_of_vector_variables_jet,
                      'weights_to_keep'                         : weights_to_keep,
                      'number_of_saved_jets'                    : number_of_saved_jets,
                      'number_of_saved_leptons'                 : number_of_saved_leptons,
                      'percentage_validation'                   : percentage_validation,
                      'mem_database_path'                       : mem_database_path,
                      'mem_database_sample_name'                : mem_database_sample_name,
                      'mem_database_sample_index_file'          : mem_database_sample_index_file,
                      }


if split_data_set:
    function_call_dict['conditions_for_splitting'] = conditions_for_splitting


NNFlow.preprocessing.root_to_HDF5(**function_call_dict)
