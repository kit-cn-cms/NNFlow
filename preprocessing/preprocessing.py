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

    if percentage_validation + percentage_test >= 100:
        sys.exit('Values for "percentage_validation" and "percentage_test" are not allowed.' + '\n')
    if not (percentage_validation > 0 and percentage_validation < 100):
        sys.exit('Value for "percentage_validation" is not allowed.' + '\n')
    if not (percentage_test >= 0 and percentage_test < 100):
        sys.exit('Value for "percentage_test" is not allowed.' + '\n')


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

    generator_level_variables = list()
    with open(path_to_generator_level_variables, 'r') as file_generator_level_variables:
        for variable in file_generator_level_variables.readlines():
            generator_level_variables.append(variable.rstrip())

    variables_to_drop = [variable for variable in generator_level_variables if variable in df.columns]
    df.drop(variables_to_drop, axis=1, inplace=True)


    #----------------------------------------------------------------------------------------------------
    # Remove variables which are not provided for all process categories.

    for variable in df.columns:
        if df[variable].isnull().any():
            df.drop(variable, axis=1, inplace=True)
            display_output.print_variable('The following variables will not be included in the training data set because they are not provided for all process categories:', variable)
    display_output.end()


    #----------------------------------------------------------------------------------------------------
    # Copy the values from the jet arrays to columns of the data frame.
    # Afterwards delete the columns which contain the arrays.

    jet_variables = [variable for variable in df.columns if isinstance(df.iloc[0].loc[variable], np.ndarray)]

    for variable in jet_variables:
        display_output.print_variable('The following variables will be treated as jet variables:', variable)

        for i in range(number_of_saved_jets):
            df[variable + '_' + str(i+1)] = df[variable].apply(lambda row: row[i] if i<len(row) else np.nan)

    df.drop(jet_variables, axis=1, inplace=True)

    display_output.end()


    #----------------------------------------------------------------------------------------------------
    # Create a boolean variable for each jet. This variable is 0 if the jet exists and 1 if the jet doesn't exist.

    for i in range(number_of_saved_jets):
        df['Jet_' + str(i+1) + '_does_not_exist'] = df['N_Jets'].apply(lambda row: 0 if i<row else 1)
        exclude_from_normalization.append('Jet_' + str(i+1) + '_does_not_exist')


    #----------------------------------------------------------------------------------------------------
    # Remove variables with standard deviation of zero.

    standard_deviation_zero_variables = list()
    for variable in df.columns:
        if variable not in process_categories:
            if df[variable].std() == 0:
                standard_deviation_zero_variables.append(variable)
                display_output.print_variable('The following variables will not be included in the training data set due to a standard deviation of zero:', variable)
            elif np.isnan(df[variable].sum()):
                standard_deviation_zero_variables.append(variable)
                display_output.print_variable('The following variables will not be included in the training data set due to a standard deviation of zero:', variable)

    df.drop(standard_deviation_zero_variables, axis=1, inplace=True)

    display_output.end()


    #----------------------------------------------------------------------------------------------------
    # Perform normalization.
    # Flags, weights and boolean jet existence variables are excluded.
    # The functions mean/std ignore np.nan values by default.
    # Mean and standard deviation are saved, so the values can be applied to the data which will be classified.

    df.apply(lambda column: column.mean() if column.name not in exclude_from_normalization else np.nan).dropna().to_msgpack(os.path.join(save_path, 'mean_normalization.msg'))
    df = df.apply(lambda column: column-column.mean() if column.name not in exclude_from_normalization else column)

    df.apply(lambda column: column.std() if column.name not in exclude_from_normalization else np.nan).dropna().to_msgpack(os.path.join(save_path, 'std_normalization.msg'))
    df = df.apply(lambda column: column/column.std() if column.name not in exclude_from_normalization else column)


    #----------------------------------------------------------------------------------------------------
    # Replace np.nan with zero for the not existing jets.

    df.fillna(value=0, inplace=True)


    #----------------------------------------------------------------------------------------------------
    # Remove all events which don't belong to the selected jet btag category.
    # Afterwards check if there are still events for each category in the data frame. If not, remove corresponding labels.

    if jet_btag_category != 'all':
        df.query(definitions.jet_btag_category(jet_btag_category), inplace=True)

        for process in process_categories[:]:
            if df[process].sum() == 0:
                df.drop(process, axis=1, inplace=True)
                process_categories.remove(process)
                display_output.print_variable("After dropping events which don't belong to the selected jet btag category, there are no events of the following processes left:", process)

        display_output.end()


    #----------------------------------------------------------------------------------------------------
    # Only keep of subset the process categories if desired.

    if selected_process_categories != 'all':
        condition = ''
        for i in range(len(selected_process_categories)):
            if i == 0:
                condition += selected_process_categories[i] + ' == 1'
            else:
                condition += ' or ' + selected_process_categories[i] + ' == 1'
        df.query(condition, inplace=True)

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
        
        df.drop(binary_classification_no_signal, axis=1, inplace=True)


    #----------------------------------------------------------------------------------------------------
    # Remove variables which have a standard deviation of zero after dropping events which don't belong to the selected jet btag category.

    standard_deviation_zero_variables_2 = list()
    for variable in df.columns:
        if variable not in process_categories:
            if df[variable].std() == 0:
                standard_deviation_zero_variables_2.append(variable)
                display_output.print_variable("The following variables will be dropped because they have a standard deviation of zero after dropping events which don't belong to the selected jet btag category:", variable)

    df.drop(standard_deviation_zero_variables_2, axis=1, inplace=True)

    display_output.end()


    #----------------------------------------------------------------------------------------------------
    # Calculate weight to be applied for the training (TODO) and remove weight variables afterwards.

    df['Training_Weight'] = 1


    for variable in df.columns:
        if variable[:6] == 'Weight':
            df.drop(variable, axis=1, inplace=True)


    #----------------------------------------------------------------------------------------------------
    # Only keep a subset of the variables if desired.

    variable_list = list()
    if select_variables=='include' or select_variables=='exclude':
        with open(path_to_variable_list, 'r') as file_variable_list:
            for variable in file_variable_list.readlines():
                variable_list.append(variable.rstrip())


    if select_variables=='include':
        unwanted_variables = [variable for variable in df.columns if variable not in variable_list]
        df.drop(unwanted_variables, axis=1, inplace=True)

    elif select_variables=='exclude':
        unwanted_variables = [variable for variable in df.columns if variable in variable_list]
        df.drop(unwanted_variables, axis=1, inplace=True)


    #----------------------------------------------------------------------------------------------------
    # Shuffle events in the data frame.
    # Afterwards split the data set into subsets for training and validation.

    df.index = np.random.permutation(df.shape[0])
    df.sort_index(inplace=True)

    number_of_validation_events = int(np.floor(percentage_validation/100*df.shape[0]))
    number_of_test_events = int(np.floor(percentage_test/100*df.shape[0]))
    number_of_training_events = df.shape[0] - number_of_validation_events - number_of_test_events

    df_train = df.head(number_of_training_events)
    df_val = df.tail(number_of_validation_events)
    if percentage_test != 0:
        df_test = df.tail(number_of_validation_events + number_of_test_events).head(number_of_test_events)


    #----------------------------------------------------------------------------------------------------
    # Save data.

    np.save(os.path.join(save_path, 'train'), df_train.values)
    np.save(os.path.join(save_path, 'val'), df_val.values)
    if percentage_test != 0:
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
