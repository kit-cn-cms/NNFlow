# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================

from __future__ import absolute_import, division, print_function

from preprocessing.preprocessing import create_dataset_for_training


#----------------------------------------------------------------------------------------------------
save_path =


#----------------------------------------------------------------------------------------------------
path_to_inputfiles =


#----------------------------------------------------------------------------------------------------
# The input data sets have to be provided as a dictionary in the following format:
# input_datasets = {'process_name':'filename'}

input_datasets =


#----------------------------------------------------------------------------------------------------
path_to_merged_data_set =


#----------------------------------------------------------------------------------------------------
path_to_weight_variables =


#----------------------------------------------------------------------------------------------------
# convert_chunksize =


#----------------------------------------------------------------------------------------------------
# The default value for the jet btag category is 'all'.
# You can choose from categories defined in definitions.definitions.jet_btag_category

#jet_btag_category =


#----------------------------------------------------------------------------------------------------
# If you don't want to keep all process categories, specify the categories you want to keep.
# All other categories will be dropped.

#selected_process_categories =


#----------------------------------------------------------------------------------------------------
# If you want to do a binary classification, specify the process which should be treated as signal.

#binary_classification        =
#binary_classification_signal =


#----------------------------------------------------------------------------------------------------
# If desired, only a subset of the available variables is saved in the data set for training.
# Option 0: All variables are kept (default).
# Option 1: Provide a list with variables you want to keep (select_variables = 'include').
# Option 2: Provide a list with variables you want to drop (select_variables = 'exclude').

#select_variables =
#path_to_variable_list =


#----------------------------------------------------------------------------------------------------
# weights_to_be_applied


#----------------------------------------------------------------------------------------------------
create_dataset_for_training(save_path                               = save_path
                          , path_to_inputfiles                      = path_to_inputfiles
                          , input_datasets                          = input_datasets
                          , path_to_merged_data_set                 = path_to_merged_data_set
                          , path_to_weight_variables                = path_to_weight_variables
#                          , convert_chunksize                       = convert_chunksize
#                          , jet_btag_category                       = jet_btag_category
#                          , selected_process_categories             = selected_process_categories
#                          , binary_classification                   = binary_classification
#                          , binary_classification_signal            = binary_classification_signal
#                          , select_variables                        = select_variables
#                          , path_to_variable_list                   = path_to_variable_list
#                          , weights_to_be_applied                   = weights_to_be_applied
                            )
