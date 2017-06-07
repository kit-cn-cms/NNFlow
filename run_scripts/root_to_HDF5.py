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
from preprocessing.preprocessing import root_to_HDF5
from definitions import definitions
#----------------------------------------------------------------------------------------------------


### Provide the file name for the output file WITHOUT file name extension.
filename_outputfile =


path_to_inputfiles =


### The filenames of the input files have to be provided as a list of strings.
filenames_inputfiles =


### The tree names have to be provided as a list of strings.
treenames =


### Here you can specify the conditions to split the data set into subprocesses.
### The conditions for splitting have to be provided as a dictionary (see example in definitions.definitions.ttbar_processes()).
split_data_set = False
#conditions_for_splitting = definitions.ttbar_processes()


#----------------------------------------------------------------------------------------------------


save_path = os.path.join(workdir_base, 'HDF5_files')

path_to_generator_level_variables       = os.path.join(NNFlow_base, 'definitions/excluded_variables/generator_level_variables.txt')
path_to_other_always_excluded_variables = os.path.join(NNFlow_base, 'definitions/excluded_variables/other_always_excluded_variables.txt')
path_to_vector_variables_lepton         = os.path.join(NNFlow_base, 'definitions/vector_variables/lepton.txt')
path_to_vector_variables_jet            = os.path.join(NNFlow_base, 'definitions/vector_variables/jet.txt')


weights_to_keep = definitions.default_weight_list()


number_of_saved_jets    = 6
number_of_saved_leptons = 1


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
                      'path_to_generator_level_variables'       : path_to_generator_level_variables,
                      'path_to_other_always_excluded_variables' : path_to_other_always_excluded_variables,
                      'path_to_vector_variables_lepton'         : path_to_vector_variables_lepton,
                      'path_to_vector_variables_jet'            : path_to_vector_variables_jet,
                      'weights_to_keep'                         : weights_to_keep,
                      'number_of_saved_jets'                    : number_of_saved_jets,
                      'number_of_saved_leptons'                 : number_of_saved_leptons,
                      'percentage_validation'                   : percentage_validation
                      }


if split_data_set:
    function_call_dict['conditions_for_splitting'] = conditions_for_splitting


root_to_HDF5(**function_call_dict)
