from __future__ import absolute_import, division, print_function

import sys


class MLP(object):

    def __init__(self,
                 savedir,
                 name,
                 number_of_input_neurons,
                 number_of_output_neurons,
                 hidden_layers,
                 activation_function,
                 ):


        self._savedir                  = savedir
        self._name                     = name

        self._number_of_input_neurons  = number_of_input_neurons
        self._number_of_output_neurons = number_of_output_neurons
        self._hidden_layers            = hidden_layers
        self._activation_function      = activation_function



    
    def _get_activation_function(self):


        activation_functions = {'tanh'    : tf.nn.tanh,
                                'sigmoid' : tf.nn.sigmoid,
                                'relu'    : tf.nn.relu,
                                'elu'     : tf.nn.elu,
                                'softplus': tf.nn.softplus
                                }

        if self._activation_function not in activation_functions.keys():
            sys.exit('Choose activation function from ' + str.join(', ', activation_functions.keys()))

        return activation_functions[self._activation_function]
