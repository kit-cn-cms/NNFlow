from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd


# Create a list of jet variables. Possibility 1: Provide the list in a file.

with open(path_to_vector_variables_jet, 'r') as file_vector_variables_jet:
    vector_variables = [variable.rstrip() for variable in file_vector_variables_jet.readlines() if variable.rstrip() in df.columns]


# Posibility 2: Check which columns contain vectors.
vector_variables = [variable for variable in df.columns if isinstance(df.iloc[0].loc[variable], np.ndarray)]


#----------------------------------------------------------------------------------------------------
# Copy the values from the arrays to columns of the data frame.
# Afterwards delete the columns which contain the arrays.

for variable in vector_variables:
    for i in range(number_of_saved_jets):
        df[variable + '_' + str(i+1)] = df[variable].apply(lambda row: row[i] if i<len(row) else np.nan)

df.drop(vector_variables_jet, axis=1, inplace=True)


#----------------------------------------------------------------------------------------------------
# Create a boolean variable for each jet. This variable is 0 if the jet exists and 1 if the jet doesn't exist.

for i in range(number_of_saved_jets):
    df['Jet_' + str(i+1) + '_does_not_exist'] = df['N_Jets'].apply(lambda row: 0 if i<row else 1)


#----------------------------------------------------------------------------------------------------
# Replace np.nan with zero for the not existing jets.

df.fillna(value=0, inplace=True)
