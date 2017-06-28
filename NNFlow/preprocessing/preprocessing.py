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
                 conditions_for_splitting={'variables': list()}):


    print('\n' + '================================')
    print(       'CONVERT ROOT FILES TO HDF5 FILES')
    print(       '================================' + '\n')


    if isinstance(filenames_inputfiles, basestring):
        filenames_inputfiles = [filenames_inputfiles]

    if isinstance(treenames, basestring):
        treenames = [treenames]

    if isinstance(weights_to_keep, basestring):
        weights_to_keep = [weights_to_keep]


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




def merge_data_sets(path_to_inputfiles,
                    input_data_sets,
                    path_to_merged_data_set,
                    columns_for_cutbased_event_selection=list()):


    print('\n' + '===============')
    print(       'MERGE DATA SETS')
    print(       '===============' + '\n')


    for filename in np.concatenate(input_data_sets.values()):
        if not os.path.isfile(os.path.join(path_to_inputfiles, filename)):
             sys.exit("File '" + os.path.join(path_to_inputfiles, filename) + "' doesn't exist." + "\n")

    if not os.path.isdir(os.path.dirname(path_to_merged_data_set)):
        sys.exit("Directory '" + os.path.dirname(path_to_merged_data_set) + "' doesn't exist." + "\n")


    #----------------------------------------------------------------------------------------------------
    # Create a list of variables that have to be dropped because they are not available in all input files.

    processes = input_data_sets.keys()

    variables_in_input_files = dict()
    for process in processes:
        for input_file in input_data_sets[process]:
            with pd.HDFStore(os.path.join(path_to_inputfiles, input_file), mode='r') as store_input:
                df = store_input.select('df_train', stop=1)
                variables_in_input_files[input_file] = set(df.columns)

    common_variables = set.intersection(*variables_in_input_files.values())

    variables_to_drop = dict()
    for process in processes:
        for input_file in input_data_sets[process]:
            variables_to_drop[input_file] = [variable for variable in variables_in_input_files[input_file] if variable not in common_variables]
            if len(variables_to_drop[input_file]) != 0:
                print("The following variables in the file '" + input_file + "' are not avialable in all input files. They will be dropped.")
                for variable in variables_to_drop[input_file]:
                    print('    ' + variable)
                print('\n', end='')


    #----------------------------------------------------------------------------------------------------
    # Merge data sets and add flags for the different processes.

    data_columns = processes + definitions.jet_btag_category()['variables'] + columns_for_cutbased_event_selection + ['Weight']

    if os.path.isfile(path_to_merged_data_set):
        os.remove(path_to_merged_data_set)

    with pd.HDFStore(path_to_merged_data_set) as store_output:
        for process in processes:
            for input_file in input_data_sets[process]:
                print('Processing ' + input_file)
                with pd.HDFStore(os.path.join(path_to_inputfiles, input_file), mode='r') as store_input:
                    for data_set in ['df_train', 'df_val', 'df_test']:
                        for df_input in store_input.select(data_set, chunksize=10000):
                            df = df_input.copy()

                            if len(variables_to_drop[input_file]) != 0:
                                df.drop(variables_to_drop[input_file], axis=1, inplace=True)
 
                            for process_label in processes:
                                df[process_label] = 1 if process_label == process else 0

                            store_output.append(data_set, df, format = 'table', append=True, data_columns=data_columns, index=False)

        print('\n', end='')
        
        for data_set in ['df_train', 'df_val', 'df_test']:
            store_output.create_table_index(data_set)

        with pd.HDFStore(os.path.join(path_to_inputfiles, input_data_sets[processes[0]][0]), mode='r') as store_input:
            weights_in_data_set = store_input.get('weights_in_data_set')
            store_output.put('weights_in_data_set', weights_in_data_set, format='fixed')

        store_output.put('processes_in_data_set', pd.Series(processes), format='fixed')


    print('\n' + '========')
    print(       'FINISHED')
    print(       '========' + '\n')




def create_data_set_for_training(save_path,
                                 path_to_merged_data_set,
                                 weights_to_be_applied,
                                 jet_btag_category,
                                 selected_processes,
                                 drop_events_negative_weights,
                                 binary_classification,
                                 select_variables,
                                 cutbased_event_selection,
                                 create_new_process_labels,
                                 binary_classification_signal=None,
                                 path_to_variable_list=None,
                                 cutbased_event_selection_condition=None,
                                 new_process_labels=None):


    print('\n' + '============================')
    print(       'CREATE DATA SET FOR TRAINING')
    print(       '============================' + '\n')
    
    
    if not os.path.isdir(save_path):
        sys.exit("Directory '" + save_path + "' doesn't exist." + "\n")

    if not os.path.isfile(path_to_merged_data_set):
        sys.exit("File '" + path_to_merged_data_set + "' doesn't exist." + "\n")
    
    if isinstance(weights_to_be_applied, basestring):
        weights_to_be_applied = [weights_to_be_applied]


    #----------------------------------------------------------------------------------------------------
    # Get processes, weights and variables in data set.
  
    with pd.HDFStore(path_to_merged_data_set, mode='r') as store:
        processes = list(store.get('processes_in_data_set').values)
        weights_in_data_set = list(store.get('weights_in_data_set').values)
        df = store.select('df_train', start=0, stop=1)
        variables_in_data_set = [variable for variable in df.columns if variable not in (processes + weights_in_data_set)]
    del df


    #----------------------------------------------------------------------------------------------------
    # Where condition.

    where_condition_list = list()

    if selected_processes != 'all':
        processes = selected_processes
        select_processes_condition = '(' + str.join(' or ', [process + ' == 1' for process in selected_processes]) + ')'
        where_condition_list.append(select_processes_condition)

    if jet_btag_category != 'all':
        jet_btag_category_condition = definitions.jet_btag_category()['conditions'][jet_btag_category]
        if not (jet_btag_category_condition[0]=='(' and jet_btag_category_condition[-1]==')'):
            jet_btag_category_condition = '(' + jet_btag_category_condition + ')'
        where_condition_list.append(jet_btag_category_condition)

    if cutbased_event_selection:
        if not (cutbased_event_selection_condition[0]=='(' and cutbased_event_selection_condition[-1]==')'):
            cutbased_event_selection_condition = '(' + cutbased_event_selection_condition + ')'
        where_condition_list.append(cutbased_event_selection_condition)

    if drop_events_negative_weights:
        where_condition_list_train = where_condition_list + ['(Weight > 0)']
    else:
        where_condition_list_train = where_condition_list


    where_condition = dict()

    if len(where_condition_list_train) == 0:
        where_condition['train'] = None
    else:
        where_condition['train'] = str.join(' and ', where_condition_list_train)

    if len(where_condition_list) == 0:
        where_condition['val'] = None
        where_condition['test'] = None
    else:
        where_condition['val'] = str.join(' and ', where_condition_list)
        where_condition['test'] = str.join(' and ', where_condition_list)


    #----------------------------------------------------------------------------------------------------
    # Make a list of variables with standard deviation of zero and a list of variables which don't exist for all events.

    print('Calculating standard deviations')
    standard_deviation_zero_variables = list()
    not_all_events_variables = list()
    with pd.HDFStore(path_to_merged_data_set, mode='r') as store:
        df_train = store.select('df_train', where=where_condition['train'], columns=variables_in_data_set)
        df_val = store.select('df_val', where=where_condition['val'], columns=variables_in_data_set)

        for variable in variables_in_data_set:
            if df_train[variable].std()==0 or df_val[variable].std()==0:
                standard_deviation_zero_variables.append(variable)

            elif df_train[variable].isnull().any() or df_val[variable].isnull().any():
                not_all_events_variables.append(variable)
        
        del df_train
        del df_val
    
    print('\n', end='')


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


    columns_to_save = list()


    if binary_classification:
        if create_new_process_labels:
            columns_to_save += new_process_labels[binary_classification_signal]
        else:
            columns_to_save.append(binary_classification_signal)
    else:
        columns_to_save += processes


    if select_variables=='include':
        columns_to_save += [variable for variable in variable_list if variable not in standard_deviation_zero_variables + not_all_events_variables]
    elif select_variables=='exclude':
        columns_to_save += [variable for variable in variables_in_data_set if variable not in standard_deviation_zero_variables + not_all_events_variables + variable_list]
    else:
        columns_to_save += [variable for variable in variables_in_data_set if variable not in standard_deviation_zero_variables + not_all_events_variables]


    #----------------------------------------------------------------------------------------------------
    # Calculate weights to be applied for the training and save data sets.

    columns_to_save_old = columns_to_save
    if create_new_process_labels:
        for new_label in new_process_labels.keys():
            processes = [new_label] + [process for process in processes if process not in new_process_labels[new_label]]
            columns_to_save = [new_label] + [column for column in columns_to_save if column not in new_process_labels[new_label]]
    
    sum_of_weights = dict()

    with pd.HDFStore(path_to_merged_data_set, mode='r') as store:
        for data_set in ['train', 'val', 'test']:
            df_weight = store.select('df_'+data_set, where=where_condition[data_set], columns=weights_to_be_applied)
            df = store.select('df_'+data_set, where=where_condition[data_set], columns=columns_to_save_old)


            if create_new_process_labels:
                for new_label in new_process_labels.keys():
                    df[new_label] = df[new_process_labels[new_label]].sum(axis=1)
                    df.drop(new_process_labels[new_label], axis=1, inplace=True)

                    df = df.reindex_axis(columns_to_save, axis=1)


            if binary_classification:
                df['Training_Weight'] = 1

                for weight in weights_to_be_applied:
                    df['Training_Weight'] *= df_weight[weight]

                if data_set == 'train':
                    sum_of_weights['signal'] = df.query(binary_classification_signal + '==1')['Training_Weight'].sum()
                    sum_of_weights['background'] = df.query(binary_classification_signal + '!=1')['Training_Weight'].sum()

                df['Training_Weight'] /= df[binary_classification_signal].apply(lambda row: sum_of_weights['signal'] if row==1 else sum_of_weights['background'])
                
                df['Training_Weight'] /= df['Training_Weight'].sum()

            else:
                df['Training_Weight'] = 1

                for weight in weights_to_be_applied:
                    df['Training_Weight'] *= df_weight[weight]
                
                if data_set == 'train':
                    for process in processes:
                        sum_of_weights[process] = df.query(process + '==1')['Training_Weight'].sum()

                for process in processes:
                    df['Training_Weight'] /= df[process].apply(lambda row: sum_of_weights[process] if row==1 else 1)

                df['Training_Weight'] /= df['Training_Weight'].sum()


            del df_weight
            np.save(os.path.join(save_path, data_set), df.values)
            del df


    #----------------------------------------------------------------------------------------------------
    # Save additional information.

    with open(os.path.join(save_path, 'variables.txt'), 'w') as outputfile_variables:
        for variable in columns_to_save:
            if variable not in processes:
                outputfile_variables.write(variable + '\n')

    if not binary_classification:
        with open(os.path.join(save_path, 'process_labels.txt'), 'w') as outputfile_process_labels:
            for process in processes:
                outputfile_process_labels.write(process + '\n')


    print('========')
    print('FINISHED')
    print('========' + '\n')