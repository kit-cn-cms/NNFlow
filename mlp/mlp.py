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




    def _get_initial_weights_biases(self):


        number_of_input_neurons  = self._number_of_input_neurons
        hidden_layers            = self._hidden_layers
        number_of_output_neurons = self._number_of_output_neurons


        weights = [tf.Variable(tf.random_normal([number_of_input_neurons, hidden_layers[0]], stddev=tf.sqrt(2.0/number_of_input_neurons)), name = 'W_1')]
        biases  = [tf.Variable(tf.fill(dims=[hidden_layers[0]], value=0.1), name = 'B_1')]


        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                weights.append(tf.Variable(tf.random_normal(shape = [hidden_layers[i-1], hidden_layers[i]], stddev = tf.sqrt(2.0/hidden_layers[i-1])), name='W_{}'.format(i+1)))
                biases.append(tf.Variable(tf.fill(dims=[hidden_layers[i]], value = 0.1), name='B_{}'.format(i+1)))


        weights.append(tf.Variable(tf.random_normal([hidden_layers[-1], number_of_output_neurons], stddev=tf.sqrt(2.0/hidden_layers[-1])), name = 'W_out'))
        biases.append(tf.Variable(tf.fill(dims=[number_of_output_neurons], value=0.1), name = 'B_out'))


        return weights, biases
