from __future__ import absolute_import, division, print_function

import NNFlow.preprocessing




try:
    import tensorflow
    del tensorflow
    tensorflow_available = True

except ImportError:
    print('\n    Warning: TensorFlow is not installed on your machine. You can only use functions that do not depend on TensorFlow.\n')
    tensorflow_available = False

if tensorflow_available:
    import NNFlow.optimizers
    from NNFlow.session_config.session_config import SessionConfig

    from NNFlow.neural_network_training.interface_functions import train_neural_network
    from NNFlow.neural_network_training.interface_functions import train_regression_network
    from NNFlow.model_analyser.binary_model_analyser        import BinaryModelAnalyser
    from NNFlow.model_analyser.multiclass_model_analyser    import MulticlassModelAnalyser


del tensorflow_available




del absolute_import
del division
del print_function
