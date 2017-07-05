from __future__ import absolute_import, division, print_function

import os
import sys

from NNFlow.preprocessing import root_to_HDF5
from NNFlow.ttH_ttbb_definitions import definitions
from NNFlow.ttH_ttbb_definitions import excluded_variables
from NNFlow.ttH_ttbb_definitions import vector_variables
#----------------------------------------------------------------------------------------------------


workdir_base =


### Provide the file name for the output file WITHOUT file name extension.
filename_outputfile = 'ttbar'


path_to_inputfiles =


### The filenames of the input files have to be provided as a list of strings.
filenames_inputfiles = sorted(os.listdir(path_to_inputfiles))


### The tree names have to be provided as a list of strings.
treenames = 'MVATree'


### Here you can specify the conditions to split the data set into subprocesses.
### The conditions for splitting have to be provided as a dictionary (see example in definitions.definitions.ttbar_processes()).
split_data_set = True
conditions_for_splitting = definitions.ttbar_processes()


#----------------------------------------------------------------------------------------------------


save_path = os.path.join(workdir_base, 'HDF5_files')

list_of_generator_level_variables       = excluded_variables.generator_level_variables()
list_of_other_always_excluded_variables = excluded_variables.other_always_excluded_variables()
list_of_vector_variables_lepton         = vector_variables.lepton_variables()
list_of_vector_variables_jet            = vector_variables.jet_variables()


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
                      'list_of_generator_level_variables'       : list_of_generator_level_variables,
                      'list_of_other_always_excluded_variables' : list_of_other_always_excluded_variables,
                      'list_of_vector_variables_lepton'         : list_of_vector_variables_lepton,
                      'list_of_vector_variables_jet'            : list_of_vector_variables_jet,
                      'weights_to_keep'                         : weights_to_keep,
                      'number_of_saved_jets'                    : number_of_saved_jets,
                      'number_of_saved_leptons'                 : number_of_saved_leptons,
                      'percentage_validation'                   : percentage_validation
                      }


if split_data_set:
    function_call_dict['conditions_for_splitting'] = conditions_for_splitting


root_to_HDF5(**function_call_dict)
