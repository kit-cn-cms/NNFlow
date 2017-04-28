# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================

from __future__ import absolute_import, division, print_function

from preprocessing.preprocessing import root_to_pandas
from definitions import definitions


#----------------------------------------------------------------------------------------------------
save_path =


#----------------------------------------------------------------------------------------------------
# Provide the file name for the output file WITHOUT file name extension.

filename_outputfile =


#----------------------------------------------------------------------------------------------------
path_to_inputfiles =


#----------------------------------------------------------------------------------------------------
# The filenames of the input files have to be provided as a list of strings.

filenames_inputfiles =


#----------------------------------------------------------------------------------------------------
# You don't have to provide tree names if each file contains exactly one tree.
# The tree names have to be provided as a list of strings.

#treenames =


#----------------------------------------------------------------------------------------------------
# The default value for 'split_data_frame' is False.

# The conditions for splitting have to be provided as a dictionary in the following format:
# conditions_for_splitting = {'process_name':'var_a == value_x and var_b != value_y or var_c > value_z'}

#split_data_frame = True
#conditions_for_splitting = definitions.ttbar_processes()


#----------------------------------------------------------------------------------------------------
root_to_pandas(save_path                = save_path
             , filename_outputfile      = filename_outputfile
             , path_to_inputfiles       = path_to_inputfiles
             , filenames_inputfiles     = filenames_inputfiles
#             , treenames                = treenames
#             , split_data_frame         = split_data_frame
#             , conditions_for_splitting = conditions_for_splitting
               )
