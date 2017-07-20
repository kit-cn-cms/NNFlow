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
    from .model_analyser.multiclass_model_analyser import MulticlassModelAnalyser
