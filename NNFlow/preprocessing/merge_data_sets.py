from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import pandas as pd

from NNFlow.definitions import definitions




def merge_data_sets(path_to_inputfiles,
                    input_data_sets,
                    path_to_merged_data_set,
                    columns_for_cutbased_event_selection=None):


    print('\n' + '===============')
    print(       'MERGE DATA SETS')
    print(       '===============' + '\n')


    for filename in np.concatenate(input_data_sets.values()):
        if not os.path.isfile(os.path.join(path_to_inputfiles, filename)):
             sys.exit("File '" + os.path.join(path_to_inputfiles, filename) + "' doesn't exist." + "\n")

    if not os.path.isdir(os.path.dirname(path_to_merged_data_set)):
        sys.exit("Directory '" + os.path.dirname(path_to_merged_data_set) + "' doesn't exist." + "\n")

    if columns_for_cutbased_event_selection is None:
        columns_for_cutbased_event_selection=list()


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
