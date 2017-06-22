from __future__ import absolute_import, division, print_function

import sys
import os
import datetime

import numpy as np

import tensorflow as tf


class MLP(object):

    
    def _get_activation_function(self,
                                 activation_function_name
                                 ):


        activation_functions = {'tanh'    : tf.nn.tanh,
                                'sigmoid' : tf.nn.sigmoid,
                                'relu'    : tf.nn.relu,
                                'elu'     : tf.nn.elu,
                                'softplus': tf.nn.softplus
                                }

        if activation_function_name not in activation_functions.keys():
            sys.exit('Choose activation function from ' + str.join(', ', activation_functions.keys()))

        return activation_functions[activation_function_name]




    def _get_initial_weights_biases(self,
                                    number_of_input_neurons,
                                    number_of_output_neurons,
                                    hidden_layers
                                    ):


        weights = [tf.Variable(tf.random_normal([number_of_input_neurons, hidden_layers[0]], stddev=tf.sqrt(2.0/number_of_input_neurons)), name = 'W_1')]
        biases  = [tf.Variable(tf.fill(dims=[hidden_layers[0]], value=0.1), name = 'B_1')]


        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                weights.append(tf.Variable(tf.random_normal(shape = [hidden_layers[i-1], hidden_layers[i]], stddev = tf.sqrt(2.0/hidden_layers[i-1])), name='W_{}'.format(i+1)))
                biases.append(tf.Variable(tf.fill(dims=[hidden_layers[i]], value = 0.1), name='B_{}'.format(i+1)))


        weights.append(tf.Variable(tf.random_normal([hidden_layers[-1], number_of_output_neurons], stddev=tf.sqrt(2.0/hidden_layers[-1])), name = 'W_out'))
        biases.append(tf.Variable(tf.fill(dims=[number_of_output_neurons], value=0.1), name = 'B_out'))


        return weights, biases




    def _get_model(self,
                   data,
                   weights,
                   biases,
                   activation_function_name,
                   keep_probability
                   ):


        activation_function = self._get_activation_function(activation_function_name)

        layers = list()
        layers.append( activation_function(tf.add(tf.matmul(data, weights[0]), biases[0])) )

        for i in range(1, len(weights-1):
            layers.append( tf.nn.dropout(activation_function(tf.add(tf.matmul(layers[i-1], weights[i]), biases[i])), keep_probability) )

        logit = tf.add(tf.matmul(layers[-1], weights[-1]), biases[-1])


        return logit




    def _get_network_and_training_properties_string(self,
                                                    number_of_input_neurons,
                                                    hidden_layers,
                                                    activation_function_name,
                                                    keep_probability,
                                                    beta,
                                                    early_stopping_interval,
                                                    batch_size,
                                                    optname,
                                                    early_stopping,
                                                    total_training_time,
                                                    mean_training_time_per_epoch
                                                    ):
        

        network_and_training_properties = str()

        network_and_training_properties += 'Date:                         {}\n'.format(datetime.datetime.now().strftime("%Y_%m_%d"))
        network_and_training_properties += 'Time:                         {}\n'.format(datetime.datetime.now().strftime("%H_%M_%S"))
        network_and_training_properties += '\n'

        network_and_training_properties += 'Number of input variables:    {}\n'.format(number_of_input_neurons)
        network_and_training_properties += '\n'

        network_and_training_properties += 'Hidden layers:                {}\n'.format(hidden_layers)
        network_and_training_properties += 'Activation function:          {}\n'.format(activation_function_name)
        network_and_training_properties += '\n'

        network_and_training_properties += 'Keep probability (dropout):   {}\n'.format(keep_probability)
        network_and_training_properties += 'L2 regularization:            {}\n'.format(beta)
        network_and_training_properties += 'Early stopping interval:      {}\n'.format(early_stopping_interval)
        network_and_training_properties += '\n'

        network_and_training_properties += 'Batch size:                   {}\n'.format(batch_size)
        network_and_training_properties += 'Optimizer:                    {}\n'.format(optname)
        network_and_training_properties += '\n'
        
        network_and_training_properties += 'Epoch early stopping:         {}\n'.format(early_stopping['epoch'])
        network_and_training_properties += '\n'

        network_and_training_properties += 'Total training time:          {} s\n'.format(total_training_time)
        network_and_training_properties += 'Mean training time per epoch: {} s\n'.format(mean_training_time_per_epoch)
        #TODO: optimizer options, best accuracy/roc, lr decay + initial

        return network_and_training_properties




    def _get_optimizer(self,
                       optimizer_name,
                       optimizer_options,
                       learning_rate,
                       decay_learning_rate,
                       decay_learning_rate_options,
                       ):


        global_step = tf.Variable(0, trainable=False)

        if decay_learning_rate:
            initial_learning_rate = learning_rate

            decay_rate  = decay_learning_rate_options['decay_rate']
            decay_steps = decay_learning_rate_options['decay_steps']

            learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_rate=decay_rate, decay_steps=decay_steps)


        if optimizer_name == "Adam":
            beta1   = optimizer_options['beta1']
            beta2   = optimizer_options['beta2']
            epsilon = optimizer_options['epsilon']

            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)


        elif optimizer_name == 'Adadelta':
            rho     = optimizer_options['rho']
            epsilon = optimizer_options['epsilon']

            optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=rho, epsilon=epsilon)


        elif optimizer_name == 'Adagrad':
            initial_accumulator_value = optimizer_options['initial_accumulator_value']

            optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=initial_accumulator_value)


        elif optimizer_name == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)


        elif optimizer_name == 'Momentum':
            momentum     = optimizer_options['momentum']
            use_nesterov = optimizer_options['use_nesterov']

            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, use_nesterov=use_nesterov)


        else:
            sys.exit('Optimizer named "{}" is not implemented.'.format(optimizer_name))


        return optimizer, global_step




    def _get_session_config(self,
                            gpu_usage
                            ):


        config = tf.ConfigProto()
        if gpu_usage['shared_machine']:
            if gpu_usage['restrict_visible_devices']:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_usage['CUDA_VISIBLE_DEVICES']

            if gpu_usage['allow_growth']:
                config.gpu_options.allow_growth = True

            if gpu_usage['restrict_per_process_gpu_memory_fraction']:
                config.gpu_options.per_process_gpu_memory_fraction = gpu_usage['per_process_gpu_memory_fraction']


        return config
