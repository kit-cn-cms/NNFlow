from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import pandas as pd

from root_numpy import root2array

from NNFlow.ttH_ttbb_definitions import definitions




def root_to_HDF5(save_path,
                 filename_outputfile,
                 path_to_inputfiles,
                 filenames_inputfiles,
                 treenames,
                 path_to_generator_level_variables,
                 path_to_other_always_excluded_variables,
                 path_to_vector_variables_lepton,
                 path_to_vector_variables_jet,
                 weights_to_keep,
                 number_of_saved_jets,
                 number_of_saved_leptons,
                 percentage_validation,
                 split_data_set,
                 conditions_for_splitting=None,
                 ):


    print('\n' + '================================')
    print(       'CONVERT ROOT FILES TO HDF5 FILES')
    print(       '================================' + '\n')


    if isinstance(filenames_inputfiles, basestring):
        filenames_inputfiles = [filenames_inputfiles]

    if isinstance(treenames, basestring):
        treenames = [treenames]

    if isinstance(weights_to_keep, basestring):
        weights_to_keep = [weights_to_keep]

    if conditions_for_splitting is None:
        conditions_for_splitting={'variables': list()}


    if not os.path.isdir(save_path):
        sys.exit("Directory '" + save_path + "' doesn't exist." + "\n")

    for filename in filenames_inputfiles:
        if not os.path.isfile(os.path.join(path_to_inputfiles, filename)):
             sys.exit("File '" + os.path.join(path_to_inputfiles, filename) + "' doesn't exist." + "\n")
   
    if not os.path.isfile(path_to_generator_level_variables):
        sys.exit("File '" + path_to_generator_level_variables + "' doesn't exist." + "\n")

    if not os.path.isfile(path_to_other_always_excluded_variables):
        sys.exit("File '" + path_to_other_always_excluded_variables + "' doesn't exist." + "\n")

    if not os.path.isfile(path_to_vector_variables_lepton):
        sys.exit("File '" + path_to_vector_variables_lepton + "' doesn't exist." + "\n")

    if not os.path.isfile(path_to_vector_variables_jet):
        sys.exit("File '" + path_to_vector_variables_jet + "' doesn't exist." + "\n")


    if not (percentage_validation > 0 and percentage_validation < 100):
        sys.exit("Value for 'percentage_validation' is not allowed." + '\n')


    #----------------------------------------------------------------------------------------------------
    # Remove old output files.

    if not split_data_set:
        if os.path.isfile(os.path.join(save_path, filename_outputfile + '.hdf')):
            os.remove(os.path.join(save_path, filename_outputfile + '.hdf'))
    else:
        for process in conditions_for_splitting['conditions'].keys():
            if os.path.isfile(os.path.join(save_path, filename_outputfile + '_' + process + '.hdf')):
                os.remove(os.path.join(save_path, filename_outputfile + '_' + process + '.hdf'))


    #----------------------------------------------------------------------------------------------------
    # Create list of generator level variables, weight variables, other always excluded variables and vector variables.

    structured_array = root2array(os.path.join(path_to_inputfiles, filenames_inputfiles[0]), treenames[0])
    df = pd.DataFrame(structured_array)

    with open(path_to_generator_level_variables, 'r') as file_generator_level_variables:
        generator_level_variables = [variable.rstrip() for variable in file_generator_level_variables.readlines() if variable.rstrip() in df.columns]
    
    with open(path_to_other_always_excluded_variables, 'r') as file_other_always_excluded_variables:
        other_excluded_variables = [variable.rstrip() for variable in file_other_always_excluded_variables.readlines() if variable.rstrip() in df.columns]

    weight_variables = [variable for variable in df.columns if variable[:6]=='Weight' and variable not in weights_to_keep]

    
    variables_for_splitting = definitions.train_test_data_set()['variables'] + conditions_for_splitting['variables']
   

    excluded_variables = [variable for variable in (generator_level_variables + weight_variables + other_excluded_variables) if variable not in variables_for_splitting]


    with open(path_to_vector_variables_lepton, 'r') as file_vector_variables_lepton:
        vector_variables_lepton = [variable.rstrip() for variable in file_vector_variables_lepton.readlines() if variable.rstrip() in df.columns and variable.rstrip() not in excluded_variables]
    with open(path_to_vector_variables_jet, 'r') as file_vector_variables_jet:
        vector_variables_jet = [variable.rstrip() for variable in file_vector_variables_jet.readlines() if variable.rstrip() in df.columns and variable.rstrip() not in excluded_variables]

    other_vector_variables = [variable for variable in df.columns if isinstance(df.iloc[0].loc[variable], np.ndarray) and variable not in vector_variables_lepton + vector_variables_jet + excluded_variables]
    excluded_variables += other_vector_variables

    del structured_array
    del df


    #----------------------------------------------------------------------------------------------------
    # Display output.

    print('Generator level variables:')
    for variable in generator_level_variables:
        print('    ' + variable)
    print('\n', end='')


    print('Dropped weight variables:')
    for variable in weight_variables:
        print('    ' + variable)
    print('\n', end='')


    print('Other excluded variables:')
    for variable in other_excluded_variables:
        print('    ' + variable)
    print('\n', end='')


    print('Lepton variables:')
    for variable in vector_variables_lepton:
        print('    ' + variable)
    print('\n', end='')


    print('Jet variables:')
    for variable in vector_variables_jet:
        print('    ' + variable)
    print('\n', end='')


    if len(other_vector_variables) != 0:
        print('The following vector variables are neither in the list of jet variables nor in the list of lepton variables. They will be removed:')
        for variable in other_vector_variables:
            print('    ' + variable)
        print('\n', end='')


    #----------------------------------------------------------------------------------------------------
    # Load and convert data.

    print('Loading and converting data')
    for filename in filenames_inputfiles:
        print('    ' + 'Processing ' + filename)
        for treename in treenames:
            structured_array = root2array(os.path.join(path_to_inputfiles, filename), treename)
            df = pd.DataFrame(structured_array)

            #--------------------------------------------------------------------------------------------
            # Remove excluded variables.
            df.drop(excluded_variables, axis=1, inplace=True)

            #--------------------------------------------------------------------------------------------
            # Copy the values from the arrays to columns of the data frame.
            for variable in vector_variables_lepton:
                for i in range(number_of_saved_leptons):
                    df[variable + '_' + str(i+1)] = df[variable].apply(lambda row: row[i] if i<len(row) else np.nan)
            df.drop(vector_variables_lepton, axis=1, inplace=True)

            for variable in vector_variables_jet:
                for i in range(number_of_saved_jets):
                    df[variable + '_' + str(i+1)] = df[variable].apply(lambda row: row[i] if i<len(row) else np.nan)
            df.drop(vector_variables_jet, axis=1, inplace=True)

            #--------------------------------------------------------------------------------------------
            # Split train, val and test data set.
            df_train = df.query(definitions.train_test_data_set()['conditions']['train']).copy()
            df_test  = df.query(definitions.train_test_data_set()['conditions']['test']).copy()

            df_train.index = np.random.permutation(df_train.shape[0])
            df_train.sort_index(inplace=True)
            number_of_validation_events = int(np.floor(percentage_validation/100*df_train.shape[0]))
            number_of_training_events = df_train.shape[0] - number_of_validation_events
            df_val = df_train.tail(number_of_validation_events).copy()
            df_train = df_train.head(number_of_training_events).copy()

            #--------------------------------------------------------------------------------------------
            # Split data set and save data.
            if not split_data_set:
                df_train.drop(variables_for_splitting, axis=1, inplace=True)
                df_val.drop(variables_for_splitting, axis=1, inplace=True)
                df_test.drop(variables_for_splitting, axis=1, inplace=True)
               
                with pd.HDFStore(os.path.join(save_path, filename_outputfile + '.hdf')) as store:
                    store.append('df_train', df_train, format = 'table', append=True)
                    store.append('df_val', df_val, format = 'table', append=True)
                    store.append('df_test', df_test, format = 'table', append=True)

            else:
                for process in conditions_for_splitting['conditions'].keys():
                    df_train_process = df_train.query(conditions_for_splitting['conditions'][process]).copy()
                    df_val_process = df_val.query(conditions_for_splitting['conditions'][process]).copy()
                    df_test_process = df_test.query(conditions_for_splitting['conditions'][process]).copy()

                    df_train_process.drop(variables_for_splitting, axis=1, inplace=True)
                    df_val_process.drop(variables_for_splitting, axis=1, inplace=True)
                    df_test_process.drop(variables_for_splitting, axis=1, inplace=True)

                    with pd.HDFStore(os.path.join(save_path, filename_outputfile + '_' + process + '.hdf')) as store:
                        store.append('df_train', df_train_process, format = 'table', append=True)
                        store.append('df_val', df_val_process, format = 'table', append=True)
                        store.append('df_test', df_test_process, format = 'table', append=True)


    #----------------------------------------------------------------------------------------------------
    # Save a list of saved weights.

    if not split_data_set:
        with pd.HDFStore(os.path.join(save_path, filename_outputfile + '.hdf')) as store:
            store.put('weights_in_data_set', pd.Series(weights_to_keep), format='fixed')
    
    else:
        for process in conditions_for_splitting['conditions'].keys():
            with pd.HDFStore(os.path.join(save_path, filename_outputfile + '_' + process + '.hdf')) as store:
                store.put('weights_in_data_set', pd.Series(weights_to_keep), format='fixed')


    print('\n' + '========')
    print(       'FINISHED')
    print(       '========' + '\n')
