from __future__ import absolute_import, division, print_function

import os
import sys
import inspect

import numpy as np
import pandas as pd

import root_numpy

from NNFlow.ttH_ttbb_definitions import definitions




def root_to_HDF5(save_path,
                 filename_outputfile,
                 path_to_inputfiles,
                 filenames_inputfiles,
                 treenames,
                 list_of_generator_level_variables,
                 list_of_other_always_excluded_variables,
                 list_of_vector_variables_lepton,
                 list_of_vector_variables_jet,
                 weights_to_keep,
                 number_of_saved_jets,
                 number_of_saved_leptons,
                 percentage_validation,
                 split_data_set,
                 conditions_for_splitting       = None,
                 mem_database_path              = None,
                 mem_database_sample_name       = None,
                 mem_database_sample_index_file = None,
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

    structured_array = root_numpy.root2array(os.path.join(path_to_inputfiles, filenames_inputfiles[0]), treenames[0])
    df = pd.DataFrame(structured_array)

    generator_level_variables = [variable for variable in list_of_generator_level_variables       if variable in df.columns]
    other_excluded_variables  = [variable for variable in list_of_other_always_excluded_variables if variable in df.columns]

    weight_variables = [variable for variable in df.columns if variable[:6]=='Weight' and variable not in weights_to_keep]

    
    variables_for_splitting = definitions.train_test_data_set()['variables'] + conditions_for_splitting['variables']
   

    excluded_variables = [variable for variable in (generator_level_variables + weight_variables + other_excluded_variables) if variable not in variables_for_splitting]


    vector_variables_lepton = [variable for variable in list_of_vector_variables_lepton if variable in df.columns and variable not in excluded_variables]
    vector_variables_jet    = [variable for variable in list_of_vector_variables_jet    if variable in df.columns and variable not in excluded_variables]

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
    # Initialize MEM database
    if mem_database_path is not None:
        import ROOT
        from NNFlow.ttH_ttbb_definitions.MEM_jet_corrections import jet_corrections

        #load library
        mem_data_base_library_path = os.path.join(os.path.dirname(inspect.getfile(sys.modules['NNFlow'])), 'MEM_database/libMEMDataBaseMEMDataBase.so')
        ROOT.gSystem.Load(mem_data_base_library_path)
      
        CvectorTString = getattr(ROOT, "std::vector<TString>")
        mem_strings_vec = CvectorTString()
        # list of the names for the mem values related to the jes/jer variations
        mem_strings=["mem_p"]
        mem_strings+=["mem_"+corr+ud+"_p" for corr in jet_corrections for ud in ["up","down"]]
        mem_strings_vec = CvectorTString()
        #print mem_strings
        # fill the string in a vector to pass to the database
        for mem_string in mem_strings:
            mem_strings_vec.push_back(ROOT.TString(mem_string))

        # initialize with path to database
        mem_data_base=ROOT.MEMDataBase(mem_database_path, mem_strings_vec )

        # load sample by identifier
        # The second argument defaults to samplename_index.txt
        # this text file simply holds a list of database files, nothing to concern you with
        mem_data_base.AddSample(mem_database_sample_name, mem_database_sample_index_file)

        #print structure of mem database
        print("MEM database:")
        mem_data_base.PrintStructure()
        print('\n', end='')
      
        # Define function to get MEM result
        def get_MEM_result(run_ID, lumi_ID, event_ID):
            result = mem_data_base.GetMEMResult(mem_database_sample_name, run_ID, lumi_ID, event_ID)
            return result.p_vec[0]


    #----------------------------------------------------------------------------------------------------
    # Load and convert data.

    print('Loading and converting data')
    for filename in filenames_inputfiles:
        print('    ' + 'Processing ' + filename)
        for treename in treenames:
            structured_array = root_numpy.root2array(os.path.join(path_to_inputfiles, filename), treename)
            df = pd.DataFrame(structured_array)

            #--------------------------------------------------------------------------------------------            
            # Assign MEM value if MEM database exists
            if mem_database_path is not None:
                df['MEM'] = df.apply(lambda row: get_MEM_result(row['Evt_Run'], row['Evt_Lumi'], row['Evt_ID']), axis=1)

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
            # Split training, validation and test data set.
            df_training_validation = df.query(definitions.train_test_data_set()['conditions']['train']).copy()
            df_test                = df.query(definitions.train_test_data_set()['conditions']['test']).copy()

            df_training_validation.index = np.random.permutation(df_training_validation.shape[0])
            df_training_validation.sort_index(inplace=True)

            number_of_validation_events = int(np.floor(percentage_validation/100*df_training_validation.shape[0]))
            number_of_training_events   = df_training_validation.shape[0] - number_of_validation_events

            df_validation = df_training_validation.tail(number_of_validation_events).copy()
            df_training   = df_training_validation.head(number_of_training_events).copy()

            del df
            del df_training_validation

            #--------------------------------------------------------------------------------------------
            # Split data set and save data.
            if not split_data_set:
                df_training.drop(variables_for_splitting, axis=1, inplace=True)
                df_validation.drop(variables_for_splitting, axis=1, inplace=True)
                df_test.drop(variables_for_splitting, axis=1, inplace=True)
               
                with pd.HDFStore(os.path.join(save_path, filename_outputfile + '.hdf')) as store:
                    store.append('df_training',   df_training,   format = 'table', append=True)
                    store.append('df_validation', df_validation, format = 'table', append=True)
                    store.append('df_test',       df_test,       format = 'table', append=True)

            else:
                for process in conditions_for_splitting['conditions'].keys():
                    df_training_process   = df_training.query(conditions_for_splitting['conditions'][process]).copy()
                    df_validation_process = df_validation.query(conditions_for_splitting['conditions'][process]).copy()
                    df_test_process       = df_test.query(conditions_for_splitting['conditions'][process]).copy()

                    df_training_process.drop(variables_for_splitting, axis=1, inplace=True)
                    df_validation_process.drop(variables_for_splitting, axis=1, inplace=True)
                    df_test_process.drop(variables_for_splitting, axis=1, inplace=True)

                    with pd.HDFStore(os.path.join(save_path, filename_outputfile + '_' + process + '.hdf')) as store:
                        store.append('df_training',   df_training_process,   format = 'table', append=True)
                        store.append('df_validation', df_validation_process, format = 'table', append=True)
                        store.append('df_test',       df_test_process,       format = 'table', append=True)


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
