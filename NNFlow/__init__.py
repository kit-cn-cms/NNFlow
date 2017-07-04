try:
    from root_numpy import root2array
    del root2array
    root2array_available = True

except ImportError:
    print '\n    Warning: You can not use the function NNFlow.preprocessing.root_to_HDF5 because root_numpy does not work on your machine.\n'
    root2array_available = False

if root2array_available:
    import NNFlow.preprocessing


try:
    import tensorflow
    del tensorflow
    tensorflow_available = True

except ImportError:
    print '\n    Warning: You can only perform the preprocessing because TensorFlow is not installed on your machine.\n'
    tensorflow_available = False

if tensorflow_available:
    import NNFlow.optimizers
    from .neural_network_training.interface_functions import train_neural_network
    from .session_config.session_config import SessionConfig
    from .model_analyser.binary_model_analyser import BinaryModelAnalyser
    from .model_analyser.onehot_model_analyser import OneHotModelAnalyser
