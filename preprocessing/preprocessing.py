from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import pandas as pd

from root_numpy import root2array

from definitions import definitions




def root_to_HDF5(save_path,
                 filename_outputfile,
                 path_to_inputfiles,
                 filenames_inputfiles,
                 path_to_generator_level_variables,
                 path_to_weight_variables,
                 path_to_other_always_excluded_variables,
                 path_to_vector_variables_lepton,
                 path_to_vector_variables_jet,
                 weights_to_keep=['Weight', 'Weight_CSV', 'Weight_PU', 'Weight_XS'],
                 treenames=[None],
                 number_of_saved_jets=6,
                 number_of_saved_leptons=1,
                 percentage_validation=20,
                 split_data_frame=False,
                 conditions_for_splitting=None,
                 variables_for_splitting=None):


    print('\n' + 'CONVERT ROOT FILES TO HDF5 FILES' + '\n')


    if isinstance(filenames_inputfiles, basestring):
        filenames_inputfiles = [filenames_inputfiles]

    if isinstance(treenames, basestring):
        treenames = [treenames]


    if not os.path.isdir(save_path):
        sys.exit("Directory " + save_path + " doesn't exist." + "\n")

    for filename in filenames_inputfiles:
        if not os.path.isfile(os.path.join(path_to_inputfiles, filename)):
             sys.exit("File " + os.path.join(path_to_inputfiles, filename) + " doesn't exist." + "\n")
   
    if not os.path.isfile(path_to_generator_level_variables):
        sys.exit("File " + path_to_generator_level_variables + " doesn't exist." + "\n")

    if not os.path.isfile(path_to_weight_variables):
        sys.exit("File " + path_to_weight_variables + " doesn't exist." + "\n")

    if not os.path.isfile(path_to_other_always_excluded_variables):
        sys.exit("File " + path_to_other_always_excluded_variables + " doesn't exist." + "\n")

    if not os.path.isfile(path_to_vector_variables_lepton):
        sys.exit("File " + path_to_vector_variables_lepton + " doesn't exist." + "\n")

    if not os.path.isfile(path_to_vector_variables_jet):
        sys.exit("File " + path_to_vector_variables_jet + " doesn't exist." + "\n")


    if not (percentage_validation > 0 and percentage_validation < 100):
        sys.exit('Value for "percentage_validation" is not allowed.' + '\n')


    #----------------------------------------------------------------------------------------------------
    # Remove old output files.

    if not split_data_frame:
        if os.path.isfile(os.path.join(save_path, filename_outputfile + '.hdf')):
            os.remove(os.path.join(save_path, filename_outputfile + '.hdf'))
    else:
        for process in conditions_for_splitting.keys():
            if os.path.isfile(os.path.join(save_path, filename_outputfile + '_' + process + '.hdf')):
                os.remove(os.path.join(save_path, filename_outputfile + '_' + process + '.hdf'))


    #----------------------------------------------------------------------------------------------------
    # Create list of generator level variables, weight variables, other always excluded variables and vector variables.

    structured_array = root2array(os.path.join(path_to_inputfiles, filenames_inputfiles[0]), treenames[0])
    df = pd.DataFrame(structured_array)

    with open(path_to_generator_level_variables, 'r') as file_generator_level_variables:
        generator_level_variables = [variable.rstrip() for variable in file_generator_level_variables.readlines() if variable.rstrip() in df.columns]
    with open(path_to_weight_variables, 'r') as file_weight_variables:
        weight_variables = [variable.rstrip() for variable in file_weight_variables.readlines() if variable.rstrip() in df.columns and variable.rstrip() not in weights_to_keep]
    with open(path_to_other_always_excluded_variables, 'r') as file_other_always_excluded_variables:
        other_excluded_variables = [variable.rstrip() for variable in file_other_always_excluded_variables.readlines() if variable.rstrip() in df.columns]
    excluded_variables = generator_level_variables + weight_variables + other_excluded_variables

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
        print('The following vector variables are neither in the lists of jet variables nor in the list of lepton variables. They will be removed:')
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
            df_train = df.query('Evt_Odd == 1').copy()
            df_test  = df.query('Evt_Odd == 0').copy()

            df_train.drop('Evt_Odd', axis=1, inplace=True)
            df_test.drop('Evt_Odd', axis=1, inplace=True)

            df_train.index = np.random.permutation(df_train.shape[0])
            df_train.sort_index(inplace=True)
            number_of_validation_events = int(np.floor(percentage_validation/100*df_train.shape[0]))
            number_of_training_events = df_train.shape[0] - number_of_validation_events
            df_val = df_train.tail(number_of_validation_events)
            df_train = df_train.head(number_of_training_events)

            #--------------------------------------------------------------------------------------------
            # Split process categories and save data.
            if not split_data_frame:
                with pd.HDFStore(os.path.join(save_path, filename_outputfile + '.hdf')) as store:
                    store.append('df_train', df_train, format = 'table', append=True)
                    store.append('df_val', df_val, format = 'table', append=True)
                    store.append('df_test', df_test, format = 'table', append=True)
            else:
                for process in conditions_for_splitting.keys():
                    df_train_process = df_train.query(conditions_for_splitting[process]).copy()
                    df_val_process = df_val.query(conditions_for_splitting[process]).copy()
                    df_test_process = df_test.query(conditions_for_splitting[process]).copy()

                    df_train_process.drop(variables_for_splitting, axis=1, inplace=True)
                    df_val_process.drop(variables_for_splitting, axis=1, inplace=True)
                    df_test_process.drop(variables_for_splitting, axis=1, inplace=True)

                    with pd.HDFStore(os.path.join(save_path, filename_outputfile + '_' + process + '.hdf')) as store:
                        store.append('df_train', df_train_process, format = 'table', append=True)
                        store.append('df_val', df_val_process, format = 'table', append=True)
                        store.append('df_test', df_test_process, format = 'table', append=True)


    print('\n' + 'FINISHED' + '\n')




def create_dataset_for_training(save_path,
                                path_to_inputfiles,
                                input_data_sets,
                                path_to_merged_data_set,
                                path_to_weight_variables,
                                convert_chunksize=10000,
                                jet_btag_category='all',
                                selected_process_categories='all',
                                binary_classification=False,
                                binary_classification_signal=None,
                                select_variables=False,
                                path_to_variable_list=None,
                                weights_to_be_applied=['Weight_PU', 'Weight_CSV']):


    print('\n' + 'CREATE DATA SET FOR TRAINING' + '\n')


    if not os.path.isdir(save_path):
        sys.exit("Directory " + save_path + " doesn't exist." + "\n")

    for filename in input_datasets.values():
        if not os.path.isfile(os.path.join(path_to_inputfiles, filename)):
             sys.exit("File " + os.path.join(path_to_inputfiles, filename) + " doesn't exist." + "\n")



    #----------------------------------------------------------------------------------------------------
    # Merge data sets and add flags for the different processes.

    process_categories = input_data_sets.keys()

    with open(path_to_weight_variables, 'r') as file_weight_variables:
        weight_variables = [variable.rstrip() for variable in file_weight_variables.readlines()]
    
    with pd.HDFStore(os.path.join(path_to_inputfiles, input_datasets[process_categories[0]]), mode='r') as store_input:
        df = store_input.select('df_train', start=0, stop=1)
        variables_in_data_set = [variable for variable in df.columns if variable not in weight_variables]

    merge_data_sets = False
    if not os.path.isfile(path_to_merged_data_set):
        merge_data_sets = True
    else:
        mtime_merged_set = os.path.getmtime(path_to_merged_data_set)
        for process in process_categories:
            if os.path.getmtime(os.path.join(path_to_inputfiles, input_datasets[process])) > mtime_merged_set:
               merge_data_sets = True

    if merge_data_sets:
        print('Merge data sets:')

        if os.path.isfile(path_to_merged_data_set):
            os.remove(path_to_merged_data_set)

        with pd.HDFStore(path_to_merged_data_set) as store_output:
            for process in process_categories:
                print('    ' + 'Processing ' + input_datasets[process])
                with pd.HDFStore(os.path.join(path_to_inputfiles, input_datasets[process]), mode='r') as store_input:
                    for data_set in ['df_train', 'df_val', 'df_test']:
                        for df_input in store_input.select(data_set, chunksize=convert_chunksize):
                            df = df_input.copy()
                            
                            for process_label in process_categories:
                                df[process_label] = 1 if process_label == process else 0

                            store_output.append(data_set, df, format = 'table', append=True, data_columns=process_categories+['N_Jets'+'N_BTagsM'])

        print('\n', end='')


    #----------------------------------------------------------------------------------------------------
    # Where condition.

    where_condition = None

    if selected_process_categories != 'all':
        process_categories = selected_process_categories
        select_process_category_condition = '(' + str.join(' or ', [process + ' == 1' for process in selected_process_categories]) + ')'

    if jet_btag_category != 'all' and selected_process_categories != 'all':
        where_condition = str.join(' and ', [definitions.jet_btag_category(jet_btag_category)] + select_process_category_condition)

    elif jet_btag_category != 'all':
        where_condition = definitions.jet_btag_category(jet_btag_category)

    elif selected_process_categories != 'all':
        where_condition = select_process_category_condition


    #----------------------------------------------------------------------------------------------------
    # Make a list of variables with standard deviation of zero and a list of variables which don't exist for all events.

    standard_deviation_zero_variables = list()
    not_all_events_variables = list()
    with pd.HDFStore(path_to_merged_data_set, mode='r') as store:
        for variable in variables_in_data_set:
            df_train = store.select('df_train', where=where_condition, columns=[variable])
            df_val = store.select('df_val', where=where_condition, columns=[variable])

            if df_train[variable].std()==0 or df_val[variable].std()==0:
                standard_deviation_zero_variables.append(variable)

            if df_train[variable].isnull().any() or df_val[variable].isnull().any():
                not_all_events_variables.append(variable)

    if len(standard_deviation_zero_variables) != 0:
        print('The following variables will be removed due to a standard deviation of zero:')
        for variable in standard_deviation_zero_variables:
            print('    ' + variable)
        print('\n', end='')

    if len(not_all_events_variables) != 0:
        print("The following variables don't exist for all events. They will be removed.")
        for variable in not_all_events_variables:
            print('    ' + variable)
        print('\n', end='')


    #----------------------------------------------------------------------------------------------------
    # Make a list of columns to be saved.

    if select_variables=='include' or select_variables=='exclude':
        with open(path_to_variable_list, 'r') as file_variable_list:
            variable_list = [variable.rstrip() for variable in file_variable_list.readlines()]


    if binary_classification:
        columns_to_save.append(binary_classification_signal)
    else:
        columns_to_save.append(process_categories)


    if select_variables=='include':
        columns_to_save += [variable for variable in variable_list if variable not in standard_deviation_zero_variables + not_all_events_variables]
    elif select_variables=='exclude':
        columns_to_save += [variable for variable in variables_in_data_set if variable not in standard_deviation_zero_variables + not_all_events_variables + variable_list]
    else:
        columns_to_save += [variable for variable in variables_in_data_set if variable not in standard_deviation_zero_variables + not_all_events_variables]


    #----------------------------------------------------------------------------------------------------
    # Calculate weights to be applied for the training and save data sets.

    sum_of_events = dict()
    with pd.HDFStore(path_to_merged_data_set, mode='r') as store:
        df_weight = store.select('df_train', where=where_condition, columns=process_categories+weights_to_be_applied)
        df = store.select('df_train', where=where_condition, columns=columns_to_save)
        
        if binary_classification:
            df_weight = store.select('df_train', where=where_condition, columns=[binary_classification_signal])

            sum_of_events['signal'] = df_weight[binary_classification_signal].sum()
            sum_of_events['background'] = df_weight.shape[0] - sum_of_events['signal']
        
        else:
            df_weight = store.select('df_train', where=where_condition, columns=process_categories)
            for process in process_categories:
                sum_of_events[process] = df_weight[process].sum()


        for data_set in ['train', 'val', 'test']:
            df_weight = store.select('df_'+data_set, where=where_condition, columns=process_categories+weights_to_be_applied)
            df = store.select('df_'+data_set, where=where_condition, columns=columns_to_save)

            if binary_classification:
                if weights_to_be_applied==None:
                    df['Trainig_Weight'] = df_weight.apply(lambda row: (1/sum_of_events['signal'] if row[binary_classification_signal] == 1 else 1/sum_of_events['background']))
                else:
                    df['Trainig_Weight'] = df_weight.apply(lambda row: row[weights_to_be_applied].product()*(1/sum_of_events['signal'] if row[binary_classification_signal] == 1 else 1/sum_of_events['background']))

            else:
                if weights_to_be_applied==None:
                    df['Trainig_Weight'] = df_weight.apply(lambda row: sum([row[process]/sum_of_events[process] for process in process_categories]))
                else:
                    df['Trainig_Weight'] = df_weight.apply(lambda row: row[weights_to_be_applied].product()*sum([row[process]/sum_of_events[process] for process in process_categories]))


            np.save(os.path.join(save_path, data_set), df.values)


    #----------------------------------------------------------------------------------------------------
    # Save additional information.

    with open(os.path.join(save_path, 'variables.txt'), 'w') as outputfile_variables:
        for variable in columns_to_save:
            if variable not in process_categories:
                outputfile_variables.write(variable + '\n')

    if not binary_classification:
        with open(os.path.join(save_path, 'process_labels.txt'), 'w') as outputfile_process_labels:
            for process in process_categories:
                if process in df_train.columns:
                    outputfile_process_labels.write(process + '\n')


    print('FINISHED' + '\n')
