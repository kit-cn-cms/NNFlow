from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import pandas as pd

from NNFlow.ttH_ttbb_definitions import definitions




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

    if not os.path.isdir(os.path.join(save_path, 'data_sets')):
        os.mkdir(os.path.join(save_path, 'data_sets'))

    if not os.path.isfile(path_to_merged_data_set):
        sys.exit("File '" + path_to_merged_data_set + "' doesn't exist." + "\n")
    
    if isinstance(weights_to_be_applied, basestring):
        weights_to_be_applied = [weights_to_be_applied]


    #----------------------------------------------------------------------------------------------------
    # Remove old output files

    if os.path.isfile(os.path.join(save_path, 'data_sets', 'training_data_set.hdf')):
        os.remove(os.path.join(save_path, 'data_sets', 'training_data_set.hdf'))

    if os.path.isfile(os.path.join(save_path, 'data_sets', 'validation_data_set.hdf')):
        os.remove(os.path.join(save_path, 'data_sets', 'validation_data_set.hdf'))

    if os.path.isfile(os.path.join(save_path, 'data_sets', 'test_data_set.hdf')):
        os.remove(os.path.join(save_path, 'data_sets', 'test_data_set.hdf'))


    #----------------------------------------------------------------------------------------------------
    # Get processes, weights and variables in data set.
  
    with pd.HDFStore(path_to_merged_data_set, mode='r') as store:
        processes = list(store.get('processes_in_data_set').values)
        weights_in_data_set = list(store.get('weights_in_data_set').values)
        df = store.select('df_training', start=0, stop=1)
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
        where_condition['training'] = None
    else:
        where_condition['training'] = str.join(' and ', where_condition_list_train)

    if len(where_condition_list) == 0:
        where_condition['validation'] = None
        where_condition['test'] = None
    else:
        where_condition['validation'] = str.join(' and ', where_condition_list)
        where_condition['test'] = str.join(' and ', where_condition_list)


    #----------------------------------------------------------------------------------------------------
    # Make a list of variables with standard deviation of zero and a list of variables which don't exist for all events.

    print('Calculating standard deviations')
    standard_deviation_zero_variables = list()
    not_all_events_variables = list()
    with pd.HDFStore(path_to_merged_data_set, mode='r') as store:
        df_training   = store.select('df_training',   where=where_condition['training'],   columns=variables_in_data_set)
        df_validation = store.select('df_validation', where=where_condition['validation'], columns=variables_in_data_set)

        for variable in variables_in_data_set:
            if df_training[variable].std()==0 or df_validation[variable].std()==0:
                standard_deviation_zero_variables.append(variable)

            elif df_training[variable].isnull().any() or df_validation[variable].isnull().any():
                not_all_events_variables.append(variable)
        
        del df_training
        del df_validation
    
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
            processes =       [new_label] + [process for process in processes       if process not in new_process_labels[new_label]]
            columns_to_save = [new_label] + [column  for column  in columns_to_save if column  not in new_process_labels[new_label]]
    
    sum_of_weights = dict()

    with pd.HDFStore(path_to_merged_data_set, mode='r') as store:
        for data_set in ['training', 'validation', 'test']:
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

                if data_set == 'training':
                    sum_of_weights['signal'] = df.query(binary_classification_signal + '==1')['Training_Weight'].sum()
                    sum_of_weights['background'] = df.query(binary_classification_signal + '!=1')['Training_Weight'].sum()

                df['Training_Weight'] /= df[binary_classification_signal].apply(lambda row: sum_of_weights['signal'] if row==1 else sum_of_weights['background'])
                
                df['Training_Weight'] /= df['Training_Weight'].sum()

            else:
                df['Training_Weight'] = 1

                for weight in weights_to_be_applied:
                    df['Training_Weight'] *= df_weight[weight]
                
                if data_set == 'training':
                    for process in processes:
                        sum_of_weights[process] = df.query(process + '==1')['Training_Weight'].sum()

                for process in processes:
                    df['Training_Weight'] /= df[process].apply(lambda row: sum_of_weights[process] if row==1 else 1)

                df['Training_Weight'] /= df['Training_Weight'].sum()


            del df_weight
            with pd.HDFStore(os.path.join(save_path, 'data_sets', data_set+'_data_set.hdf')) as store_output:
                store_output.put('data', df, format='fixed')
                store_output.put('variables', pd.Series([variable for variable in columns_to_save if variable not in processes]), format='fixed')
                if not binary_classification:
                    store_output.put('processes', pd.Series(processes), format='fixed')
            del df


    #----------------------------------------------------------------------------------------------------
    # Save additional information.

    with open(os.path.join(save_path, 'inputVariables.txt'), 'w') as outputfile_variables:
        for variable in columns_to_save:
            if variable not in processes:
                outputfile_variables.write(variable + '\n')

    if not binary_classification:
        with open(os.path.join(save_path, 'outputLabels.txt'), 'w') as outputfile_process_labels:
            for process in processes:
                outputfile_process_labels.write(process + '\n')


    print('========')
    print('FINISHED')
    print('========' + '\n')
