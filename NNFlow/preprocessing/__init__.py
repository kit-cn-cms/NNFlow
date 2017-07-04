try:
    from root_numpy import root2array
    del root2array
    root2array_available = True

except ImportError:
    print '\n    Warning: You can not use the function NNFlow.preprocessing.root_to_HDF5 because root_numpy does not work on your machine.\n'
    root2array_available = False

if root2array_available:
    from .root_to_HDF5 import root_to_HDF5


from .merge_data_sets              import merge_data_sets
from .create_data_set_for_training import create_data_set_for_training
