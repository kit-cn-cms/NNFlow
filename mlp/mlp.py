from __future__ import absolute_import, division, print_function

import os
import sys
import time
import datetime
import itertools

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from mlp.dataframe import DataFrame


class MLP(object):


    def train(self,
              save_path,
              model_name,
              number_of_input_neurons,
              number_of_output_neurons,
              hidden_layers,
              activation_function_name,
              dropout_keep_probability,
              l2_regularization_beta,
              early_stopping_intervall,
              path_to_training_data_set,
              path_to_validation_data_set,
              batch_size,
              optimizer_options,
              gpu_usage
              ):
 
 
        print('\n' + 'TRAINING BINARY MLP' + '\n')
 
 
        if not os.path.isdir(save_path):
            sys.exit("Directory '" + save_path + "' doesn't exist." + "\n")
 
        if not os.path.isdir(path_to_training_data_set):
            sys.exit("Directory '" + path_to_training_data_set + "' doesn't exist." + "\n")
 
        if not os.path.isdir(path_to_validation_data_set:
            sys.exit("Directory '" + path_to_training_data_set + "' doesn't exist." + "\n")
 
 
        path_to_model_file = os.path.join(save_path, '/{}.ckpt'.format(model_name))
 
 
        training_data_set   = DataFrame(path_to_training_data_set, number_of_output_neurons)
        validation_data_set = DataFrame(path_to_validation_data_set, number_of_output_neurons)
 
 
        graph = tf.Graph()
        with graph.as_default():
            input_data    = tf.placeholder(tf.float32, [None, number_of_input_neurons], name='input')
            labels        = tf.placeholder(tf.float32, [None])
            event_weights = tf.placeholder(tf.float32, [None])
 
            feature_scaling_mean = tf.Variable(np.mean(training_data.x, axis=0).astype(np.float32), trainable=False,  name='feature_scaling_mean')
            feature_scaling_std  = tf.Variable(np.std(training_data.x, axis=0).astype(np.float32), trainable=False,  name='feature_scaling_std')
            input_data_scaled    = tf.div(tf.subtract(input_data, feature_scaling_mean), feature_scaling_std)
 
            weights, biases = self._get_initial_weights_biases(number_of_input_neurons, number_of_output_neurons, hidden_layers)
 
            logits     =               tf.reshape(self._get_model(input_data_scaled, weights, biases, activation_function_name, dropout_keep_probability=dropout_keep_probability), [-1])
            prediction = tf.nn.sigmoid(tf.reshape(self._get_model(input_data_scaled, weights, biases, activation_function_name, dropout_keep_probability=1), [-1]), name='output')

            cross_entropy     = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=event_labels)
            l2_regularization = beta * tf.add_n([tf.nn.l2_loss(w) for w in weights])
            loss              = tf.add(tf.reduce_mean(tf.multiply(event_weights, cross_entropy)), l2_regularization)
 
            optimizer, global_step = self._get_optimizer(optimizer_options)
            train_step             = optimizer.minimize(loss, global_step=global_step)
 
            saver = tf.train.Saver(weights + biases + [feature_scaling_mean, feature_scaling_std])
 
 
        config = self._get_session_config(gpu_usage)
        with tf.Session(config=config, graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
 
 
            training_roc_auc   = list()
            validation_roc_auc = list()
            training_losses    = list()
            early_stopping     = {'auc': -1.0, 'epoch': 0}
            epoch_durations    = list()
 
 
            print(100*'-')
            print('{:^25} | {:^25} | {:^25} |{:^25}'.format('Epoch', 'Training data: loss', 'Training data: ROC AUC', 'Validation data: ROC AUC'))
            print(100*'-')
 
            for epoch in itertools.count(start=1, step=1)
                epoch_start = time.time()
 
                for batch_data, batch_labels, batch_event_weights in training_data_set.batches(batch_size):
                    sess.run(train_step, {input_data    : batch_data,
                                          labels        : batch_labels,
                                          event_weights : batch_event_weights})
                    
                # monitor training
                batch_predictions = list
                batch_losses      = list()
                for batch_data, batch_labels, batch_event_weights in training_data_set.batches(batch_size):
                    batch_prediction, batch_loss = sess.run([prediction, loss], {input_data    : batch_data,
                                                                                 labels        : batch_labels,
                                                                                 event_weights : batch_event_weights})
 
                    batch_predictions.append(batch_prediction)
                    batch_losses.append(batch_loss)
 
                training_prediction = np.concatenate(batch_predictions, axis=0)
                train_auc.append(roc_auc_score(y_true = train_data.y[:total_batches*batch_size], y_score = train_pre, sample_weight = train_data.w[:total_batches*batch_size]))
                validation_prediction = sess.run(prediction, {input_data : validation_data_set.get_data()})
                validation_roc_auc.append(roc_auc_score(y_true = val_data.y, y_score = val_pre, sample_weight = val_data.w))
                
                print('{:^25} | {:^25.4e} | {:^25.4f} | {:^25.4f}'.format(epoch, training_losses[-1], train_auc[-1], val_auc[-1]))
 
                # check for early stopping, only save model if val_auc has increased
                if val_auc[-1] > early_stopping['auc']:
                    saver.save(sess, path_to_model_file)
 
                    early_stopping['auc']     = val_auc[-1]
                    early_stopping['epoch']   = epoch
 
 
                elif (epoch - early_stopping['epoch']) >= early_stop:
                    print(100*'-')
                    print('Validation AUC has not increased for {} epochs. Achieved best validation auc score of {:.4f} in epoch {}'.format(early_stop, early_stopping['auc'], early_stopping['epoch']))
                    break
                
                
                epoch_end = time.time()
                epoch_durations.append(epoch_end - epoch_start)
 
 
            print(100*'-')
            network_and_training_properties_string = self._get_network_and_training_properties_string()
            print(network_and_training_properties_string)
            print(100*'-' + '\n')




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
                   dropout_keep_probability
                   ):


        activation_function = self._get_activation_function(activation_function_name)

        layers = list()
        layers.append( activation_function(tf.add(tf.matmul(data, weights[0]), biases[0])) )

        for i in range(1, len(weights-1):
            layers.append( tf.nn.dropout(activation_function(tf.add(tf.matmul(layers[i-1], weights[i]), biases[i])), dropout_keep_probability) )

        logit = tf.add(tf.matmul(layers[-1], weights[-1]), biases[-1])


        return logit




    def _get_network_and_training_properties_string(self,
                                                    number_of_input_neurons,
                                                    hidden_layers,
                                                    activation_function_name,
                                                    dropout_keep_probability,
                                                    l2_regularization_beta,
                                                    early_stopping_interval,
                                                    batch_size,
                                                    optimizer_options,
                                                    early_stopping,
                                                    total_training_time,
                                                    mean_training_time_per_epoch
                                                    ):
        

        network_and_training_properties = str()


        network_and_training_properties += 'Date:                         {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d"))
        network_and_training_properties += 'Time:                         {}\n'.format(datetime.datetime.now().strftime("%H:%M:%S"))
        network_and_training_properties += '\n'


        network_and_training_properties += 'Number of input variables:    {}\n'.format(number_of_input_neurons)
        network_and_training_properties += '\n'


        network_and_training_properties += 'Hidden layers:                {}\n'.format(hidden_layers)
        network_and_training_properties += 'Activation function:          {}\n'.format(activation_function_name)
        network_and_training_properties += '\n'


        network_and_training_properties += 'Keep probability (dropout):   {}\n'.format(dropout_keep_probability)
        network_and_training_properties += 'L2 regularization:            {}\n'.format(l2_regularization_beta)
        network_and_training_properties += 'Early stopping interval:      {}\n'.format(early_stopping_interval)
        network_and_training_properties += '\n'


        network_and_training_properties += 'Batch size:                   {}\n'.format(batch_size)
        network_and_training_properties += 'Optimizer:                    {}\n'.format(optimizer_options['name'])

        optimizer_options_keys     = [key for key in optimizer_options.keys() if 'learning_rate' not in key and key!='name'].sort()
        learning_rate_options_keys = [key for key in optimizer_options.keys() if 'learning_rate' in key].sort()

        for key in optimizer_options_keys:
            network_and_training_properties += '{:30}{}\n'.format(key, optimizer_options[key])

        for key in learning_rate_options_keys:
            network_and_training_properties += '{:30}{}\n'.format(key, optimizer_options[key])

        network_and_training_properties += '\n'


        network_and_training_properties += 'Epoch early stopping:         {}\n'.format(early_stopping['epoch'])
        network_and_training_properties += '\n'


        network_and_training_properties += 'Mean training time per epoch: {} s\n'.format(mean_training_time_per_epoch)
        #TODO: best accuracy/roc

        return network_and_training_properties




    def _get_optimizer(self,
                       optimizer_options,
                       ):


        global_step = tf.Variable(0, trainable=False)

        if optimizer_options['learning_rate_decay']:
            initial_learning_rate = optimizer_options['learning_rate_decay_initial_value']

            decay_rate  = optimizer_options['learning_rate_decay_rate']
            decay_steps = optimizer_options['learning_rate_decay_steps']

            learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_rate=decay_rate, decay_steps=decay_steps)

        else:
            learning_rate = learning_rate_options['learning_rate']


        if optimizer_options['name'] == "Adam":
            beta1   = optimizer_options['beta1']
            beta2   = optimizer_options['beta2']
            epsilon = optimizer_options['epsilon']

            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)


        elif optimizer_options['name'] == 'Adadelta':
            rho     = optimizer_options['rho']
            epsilon = optimizer_options['epsilon']

            optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=rho, epsilon=epsilon)


        elif optimizer_options['name'] == 'Adagrad':
            initial_accumulator_value = optimizer_options['initial_accumulator_value']

            optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=initial_accumulator_value)


        elif optimizer_options['name'] == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)


        elif optimizer_options['name'] == 'Momentum':
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
