from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import pandas as pd

from root_numpy import root2array

from definitions import definitions


class variable_printer:
    def __init__(self):
        self._printed_heading = False

    def print_variable(self, heading, variable):
        if not self._printed_heading:
            print(heading)
            self._printed_heading = True
        print('    ' + variable)

    def end(self):
        if self._printed_heading:
            print('\n', end='')
        self._printed_heading = False


def root_to_pandas(save_path,
                   filename_outputfile,
                   path_to_inputfiles,
                   filenames_inputfiles,
                   treenames=[None],
                   split_data_frame=False,
                   conditions_for_splitting=None):


    print('\n' + 'CONVERT ROOT FILES TO PANDAS DATA FRAMES' + '\n')


    if isinstance(filenames_inputfiles, basestring):
        filenames_inputfiles = [filenames_inputfiles]

    if isinstance(treenames, basestring):
        treenames = [treenames]


    if not os.path.isdir(save_path):
        sys.exit("Directory " + save_path + " doesn't exist." + "\n")

    for filename in filenames_inputfiles:
        if not os.path.isfile(os.path.join(path_to_inputfiles, filename)):
             sys.exit("File " + os.path.join(path_to_inputfiles, filename) + " doesn't exist." + "\n")


    print('Loading and converting data')
    df_list = list()
    for filename in filenames_inputfiles:
        print('    ' + 'Processing ' + filename)
        for treename in treenames:
            structured_array = root2array(os.path.join(path_to_inputfiles, filename), treename)
            df = pd.DataFrame(structured_array)
            df_list.append(df)

    df = pd.concat(df_list)


    print('\n' + 'Saving data')
    if not split_data_frame:
        pd.to_msgpack(os.path.join(save_path, filename_outputfile) + '.msg', np.array_split(df, int(np.ceil(df.shape[0]/400000))))

    else:
        for process in conditions_for_splitting.keys():
            df_process = df.query(conditions_for_splitting[process])
            pd.to_msgpack(os.path.join(save_path, filename_outputfile) + '_' + process + '.msg', np.array_split(df_process, int(np.ceil(df_process.shape[0]/400000))))


    print('\n' + 'FINISHED' + '\n')




def create_dataset_for_training(save_path,
                                path_to_inputfiles,
                                input_datasets,
                                path_to_generator_level_variables,
                                path_to_weight_variables,
                                path_to_other_always_excluded_variables,
                                path_to_vector_variables_first_entry,
                                path_to_vector_variables_jet,
                                number_of_saved_jets=10, 
                                jet_btag_category='all',
                                selected_process_categories='all',
                                binary_classification=False,
                                binary_classification_signal=None,
                                select_variables=False,
                                path_to_variable_list=None,
                                percentage_validation=5):


    print('\n' + 'CREATE DATA SET FOR TRAINING' + '\n')


    if not os.path.isdir(save_path):
        sys.exit("Directory " + save_path + " doesn't exist." + "\n")

    for filename in input_datasets.values():
        if not os.path.isfile(os.path.join(path_to_inputfiles, filename)):
             sys.exit("File " + os.path.join(path_to_inputfiles, filename) + " doesn't exist." + "\n")

    if not (percentage_validation > 0 and percentage_validation < 100):
        sys.exit('Value for "percentage_validation" is not allowed.' + '\n')


    display_output = variable_printer()


    #----------------------------------------------------------------------------------------------------
    # Load data, add flags for the different processes and concatenate the data frames.

    df_list = list()
    process_categories = input_datasets.keys()

    print('Load data:')
    for process in process_categories:
        print('    ' + 'Loading ' + input_datasets[process])
        df = pd.concat(pd.read_msgpack(os.path.join(path_to_inputfiles, input_datasets[process])))

        columns_old = df.columns
        columns_new = np.concatenate([process_categories, columns_old])

        for process_label in process_categories:
            df[process_label] = 1 if process_label == process else 0

        df = df.reindex_axis(columns_new, axis=1)

        df_list.append(df)

    df = pd.concat(df_list)
    print('\n', end='')

    exclude_from_normalization = list()
    exclude_from_normalization += process_categories


    #----------------------------------------------------------------------------------------------------
    # Remove generator level variables.

    with open(path_to_generator_level_variables, 'r') as file_generator_level_variables:
        generator_level_variables = [variable.rstrip() for variable in file_generator_level_variables.readlines() if variable.rstrip() in df.columns]

    df.drop(generator_level_variables, axis=1, inplace=True)


    #----------------------------------------------------------------------------------------------------
    # Remove other always excluded variables.

    with open(path_to_other_always_excluded_variables, 'r') as file_other_always_excluded_variables:
        other_always_excluded_variables = [variable.rstrip() for variable in file_other_always_excluded_variables.readlines() if variable.rstrip() in df.columns]

    df.drop(other_always_excluded_variables, axis=1, inplace=True)


    #----------------------------------------------------------------------------------------------------
    # Remove variables which are not provided for all process categories.

    for variable in df.columns:
        if df[variable].isnull().any():
            df.drop(variable, axis=1, inplace=True)
            display_output.print_variable('The following variables will not be included in the training data set because they are not provided for all process categories:', variable)
    display_output.end()


    #----------------------------------------------------------------------------------------------------
    # Copy the values from the arrays to columns of the data frame.
    # Afterwards delete the columns which contain the arrays.

    with open(path_to_vector_variables_first_entry, 'r') as file_vector_variables_first_entry:
        vector_variables_first_entry = [variable.rstrip() for variable in file_vector_variables_first_entry.readlines() if variable.rstrip() in df.columns]

    with open(path_to_vector_variables_jet, 'r') as file_vector_variables_jet:
        vector_variables_jet = [variable.rstrip() for variable in file_vector_variables_jet.readlines() if variable.rstrip() in df.columns]


    for variable in vector_variables_first_entry:
        display_output.print_variable('For the following vector variables only the first entry will be saved:', variable)
        df[variable + '_' + str(1)] = df[variable].apply(lambda row: row[0])

    df.drop(vector_variables_first_entry, axis=1, inplace=True)

    display_output.end()


    for variable in vector_variables_jet:
        display_output.print_variable('The following variables will be treated as jet variables:', variable)

        for i in range(number_of_saved_jets):
            df[variable + '_' + str(i+1)] = df[variable].apply(lambda row: row[i] if i<len(row) else np.nan)

    df.drop(vector_variables_jet, axis=1, inplace=True)

    display_output.end()


    other_vector_variables = [variable for variable in df.columns if isinstance(df.iloc[0].loc[variable], np.ndarray)]
    if len(other_vector_variables):
        for variable in other_vector_variables:
            display_output.print_variable("You didn't specify what to do with the following vector variables. They will be dropped.", variable)
        display_output.end()

        df.drop(other_vector_variables, axis=1, inplace=True)

    #----------------------------------------------------------------------------------------------------
    # Create a boolean variable for each jet. This variable is 0 if the jet exists and 1 if the jet doesn't exist.

    for i in range(number_of_saved_jets):
        df['Jet_' + str(i+1) + '_does_not_exist'] = df['N_Jets'].apply(lambda row: 0 if i<row else 1)
        exclude_from_normalization.append('Jet_' + str(i+1) + '_does_not_exist')


    #----------------------------------------------------------------------------------------------------
    # Split into training and test data set.

    df_train = df.query('Evt_Odd == 1')
    df_test  = df.query('Evt_Odd == 0')

    df_train.drop('Evt_Odd', axis=1, inplace=True)
    df_test.drop('Evt_Odd', axis=1, inplace=True)

    del df


    #----------------------------------------------------------------------------------------------------
    # Remove variables with standard deviation of zero.

    standard_deviation_zero_variables = list()
    for variable in df_train.columns:
        if variable not in process_categories:
            if (df_train[variable].std() == 0) or (df_train[variable].isnull().all()) or (np.isnan(df_train[variable].std())):
                standard_deviation_zero_variables.append(variable)
                display_output.print_variable('The following variables will not be included in the training data set due to a standard deviation of zero:', variable)

    df_train.drop(standard_deviation_zero_variables, axis=1, inplace=True)
    df_test.drop(standard_deviation_zero_variables, axis=1, inplace=True)

    display_output.end()


    #----------------------------------------------------------------------------------------------------
    # Perform normalization. TODO: Test Sample / Save values / Delete section?
    # Flags, weights and boolean jet existence variables are excluded.
    # The functions mean/std ignore np.nan values by default.
    # Mean and standard deviation are saved, so the values can be applied to the data which will be classified.

#    df.apply(lambda column: column.mean() if column.name not in exclude_from_normalization else np.nan).dropna().to_msgpack(os.path.join(save_path, 'mean_normalization.msg'))
    df_train = df_train.apply(lambda column: column-column.mean() if column.name not in exclude_from_normalization else column)

#    df.apply(lambda column: column.std() if column.name not in exclude_from_normalization else np.nan).dropna().to_msgpack(os.path.join(save_path, 'std_normalization.msg'))
    df_train = df_train.apply(lambda column: column/column.std() if column.name not in exclude_from_normalization else column)


    #----------------------------------------------------------------------------------------------------
    # Replace np.nan with zero for the not existing jets.

    df_train.fillna(value=0, inplace=True)


    #----------------------------------------------------------------------------------------------------
    # Remove all events which don't belong to the selected jet btag category.
    # Afterwards check if there are still events for each category in the data frame. If not, remove corresponding labels.

    if jet_btag_category != 'all':
        df_train.query(definitions.jet_btag_category(jet_btag_category), inplace=True)
        df_test.query(definitions.jet_btag_category(jet_btag_category), inplace=True)

        for process in process_categories[:]:
            if df_train[process].sum() == 0:
                df_train.drop(process, axis=1, inplace=True)
                process_categories.remove(process)

                if df_test[process].sum != 0:
                    df_test.query(process + '== 0', inplace=True)

                df_test.drop(process, axis=1, inplace=True)

                display_output.print_variable("After dropping events which don't belong to the selected jet btag category, there are no events of the following processes left:", process)

        display_output.end()


    #----------------------------------------------------------------------------------------------------
    # Only keep a subset of the process categories if desired.

    if selected_process_categories != 'all':
        condition = ''
        for i in range(len(selected_process_categories)):
            if i == 0:
                condition += selected_process_categories[i] + ' == 1'
            else:
                condition += ' or ' + selected_process_categories[i] + ' == 1'
        df_train.query(condition, inplace=True)
        df_test.query(condition, inplace=True)

        for process in process_categories[:]:
            if df[process].sum() == 0:
                df.drop(process, axis=1, inplace=True)
                process_categories.remove(process)
                display_output.print_variable("The following process categories have been dropped:", process)

        display_output.end()


    #----------------------------------------------------------------------------------------------------
    # Adjust process labels for binary classification if desired.

    if binary_classification:
        binary_classification_no_signal = [process for process in process_categories if process != binary_classification_signal]
        
        df_train.drop(binary_classification_no_signal, axis=1, inplace=True)
        df_test.drop(binary_classification_no_signal, axis=1, inplace=True)


    #----------------------------------------------------------------------------------------------------
    # Remove variables which have a standard deviation of zero after dropping events which don't belong to the selected jet btag category.

    standard_deviation_zero_variables_2 = list()
    for variable in df_train.columns:
        if variable not in process_categories:
            if df_train[variable].std() == 0:
                standard_deviation_zero_variables_2.append(variable)
                display_output.print_variable("The following variables will be dropped because they have a standard deviation of zero after dropping events which don't belong to the selected jet btag category:", variable)

    df_train.drop(standard_deviation_zero_variables_2, axis=1, inplace=True)
    df_test.drop(standard_deviation_zero_variables_2, axis=1, inplace=True)

    display_output.end()


    #----------------------------------------------------------------------------------------------------
    # Calculate weight to be applied for the training and remove weight variables afterwards.
    # TODO: Apply other weights, mechanism for multinomial classification.

    training_weights = dict()
    if binary_classification:
        N_signal = df_train.iloc[:, 0].sum()
        N_background = df_train.shape[0] - N_signal
        
        training_weights['signal'] = 1/N_signal
        training_weights['background'] = 1/N_background

        df_train['Training_Weight'] = df_train.iloc[:, 0].apply(lambda row: training_weights['signal'] if row==1 else training_weights['background'])

    else:
        df_train['Training_Weight'] = 1


    with open(path_to_weight_variables, 'r') as file_weight_variables:
        weight_variables = [variable.rstrip() for variable in file_weight_variables.readlines() if variable.rstrip() in df.columns]

    df_train.drop(weight_variables, axis=1, inplace=True)
    df_test.drop(weight_variables, axis=1, inplace=True)


    #----------------------------------------------------------------------------------------------------
    # Only keep a subset of the variables if desired.

    if select_variables=='include' or select_variables=='exclude':
        with open(path_to_variable_list, 'r') as file_variable_list:
            variable_list = [variable.rstrip() for variable in file_variable_list.readlines()]


    if select_variables=='include':
        unwanted_variables = [variable for variable in df.columns if variable not in variable_list]
        df_train.drop(unwanted_variables, axis=1, inplace=True)
        df_test.drop(unwanted_variables, axis=1, inplace=True)

    elif select_variables=='exclude':
        unwanted_variables = [variable for variable in df.columns if variable in variable_list]
        df_train.drop(unwanted_variables, axis=1, inplace=True)
        df_test.drop(unwanted_variables, axis=1, inplace=True)


    #----------------------------------------------------------------------------------------------------
    # Shuffle events in the data frame.
    # Afterwards split the data set into subsets for training and validation.

    df_train.index = np.random.permutation(df.shape[0])
    df_train.sort_index(inplace=True)

    number_of_validation_events = int(np.floor(percentage_validation/100*df_train.shape[0]))
    number_of_training_events = df_train.shape[0] - number_of_validation_events

    df_val = df_train.tail(number_of_validation_events)
    df_train = df_train.head(number_of_training_events)


    #----------------------------------------------------------------------------------------------------
    # Save data.

    np.save(os.path.join(save_path, 'train'), df_train.values)
    np.save(os.path.join(save_path, 'val'), df_val.values)
    np.save(os.path.join(save_path, 'test'), df_test.values)

    with open(os.path.join(save_path, 'variables.txt'), 'w') as outputfile_variables:
        for variable in df.columns:
            if variable not in process_categories and variable != 'Training_Weight':
                outputfile_variables.write(variable + '\n')

    with open(os.path.join(save_path, 'process_labels.txt'), 'w') as outputfile_process_labels:
        for process in process_categories:
            if process in df.columns:
                outputfile_process_labels.write(process + '\n')


    print('FINISHED' + '\n')
