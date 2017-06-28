from __future__ import absolute_import, division, print_function

import os
import sys
import time
import datetime
import itertools

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from data_frame.data_frame import DataFrame


class MLP(object):


    def train(self,
              save_path,
              model_name,
              network_type,
              number_of_output_neurons,
              hidden_layers,
              activation_function_name,
              dropout_keep_probability,
              l2_regularization_beta,
              early_stopping_intervall,
              path_to_training_data_set,
              path_to_validation_data_set,
              optimizer_options,
              batch_size_training,
              batch_size_classification,
              gpu_usage
              ):


        print('\n' + '============')
        print(       'TRAINING MLP')
        print(       '============' + '\n')

 
        if not os.path.isdir(save_path):
            sys.exit("Directory '" + save_path + "' doesn't exist." + "\n")
 
        if not os.path.isfile(path_to_training_data_set):
            sys.exit("File '" + path_to_training_data_set + "' doesn't exist." + "\n")
 
        if not os.path.isfile(path_to_validation_data_set):
            sys.exit("File '" + path_to_training_data_set + "' doesn't exist." + "\n")
 
 
        path_to_model_file = os.path.join(save_path, '{}.ckpt'.format(model_name))


        #----------------------------------------------------------------------------------------------------
        # Load data.
 
        training_data_set   = DataFrame(path_to_training_data_set, number_of_output_neurons)
        validation_data_set = DataFrame(path_to_validation_data_set, number_of_output_neurons)

        number_of_input_neurons = training_data_set.get_number_of_variables()


        #----------------------------------------------------------------------------------------------------
        # Graph.

        graph = tf.Graph()
        with graph.as_default():
            input_data    = tf.placeholder(tf.float32, [None, number_of_input_neurons], name='input')
            event_weights = tf.placeholder(tf.float32, [None])
 
            feature_scaling_mean = tf.Variable(np.mean(training_data_set.get_data(), axis=0).astype(np.float32), trainable=False,  name='feature_scaling_mean')
            feature_scaling_std  = tf.Variable(np.std(training_data_set.get_data(), axis=0).astype(np.float32), trainable=False,  name='feature_scaling_std')
            input_data_scaled    = tf.div(tf.subtract(input_data, feature_scaling_mean), feature_scaling_std)
 
            weights, biases = self._get_initial_weights_biases(number_of_input_neurons, number_of_output_neurons, hidden_layers)


            if network_type == 'binary':
                labels = tf.placeholder(tf.float32, [None])

                logits      =               tf.reshape(self._get_model(input_data_scaled, weights, biases, activation_function_name, dropout_keep_probability=dropout_keep_probability), [-1])
                predictions = tf.nn.sigmoid(tf.reshape(self._get_model(input_data_scaled, weights, biases, activation_function_name, dropout_keep_probability=1), [-1]), name='output')

                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)


            elif network_type == 'one-hot':
                labels = tf.placeholder(tf.float32, [None, number_of_output_neurons])

                logits      =               self._get_model(input_data_scaled, weights, biases, activation_function_name, dropout_keep_probability=dropout_keep_probability)
                predictions = tf.nn.softmax(self._get_model(input_data_scaled, weights, biases, activation_function_name, dropout_keep_probability=1), name='output')

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)


            l2_regularization = l2_regularization_beta * tf.add_n([tf.nn.l2_loss(w) for w in weights])
            loss              = tf.add(tf.reduce_mean(tf.multiply(event_weights, cross_entropy)), l2_regularization)
 
            optimizer, global_step = self._get_optimizer_global_step(optimizer_options)
            train_step             = optimizer.minimize(loss, global_step=global_step)
 
            saver = tf.train.Saver(weights + biases + [feature_scaling_mean, feature_scaling_std])
 

        #----------------------------------------------------------------------------------------------------

        config = self._get_session_config(gpu_usage)
        with tf.Session(config=config, graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
 
 
            training_accuracies   = list()
            validation_accuracies = list()
            training_losses       = list()
            early_stopping        = {'validation_accuracy': -1.0, 'epoch': 0}
            epoch_durations       = list()


            print('\n', end='')
            print(110*'-')
            if network_type == 'binary':
                print('{:^25} | {:^25} | {:^25} | {:^25}'.format('Epoch', 'Training data: loss', 'Training data: ROC AUC', 'Validation data: ROC AUC'))
            elif network_type == 'one-hot':
                print('{:^25} | {:^25} | {:^25} | {:^25}'.format('Epoch', 'Training data: loss', 'Training data: accuracy', 'Validation data: accuracy'))
            print(110*'-')


            for epoch in itertools.count(start=1, step=1):
                epoch_start = time.time()


                #----------------------------------------------------------------------------------------------------
                # Loop over batches and minimize loss.

                for batch_data, batch_labels, batch_event_weights in training_data_set.get_data_labels_event_weights_as_batches(batch_size                 = batch_size_training,
                                                                                                                                sort_events_randomly       = True,
                                                                                                                                include_smaller_last_batch = False
                                                                                                                                ):
                    sess.run(train_step, {input_data    : batch_data,
                                          labels        : batch_labels,
                                          event_weights : batch_event_weights})


                #----------------------------------------------------------------------------------------------------
                # Make predictions for training data set.

                training_batch_predictions = list()
                training_batch_losses      = list()

                for batch_data, batch_labels, batch_event_weights in training_data_set.get_data_labels_event_weights_as_batches(batch_size                 = batch_size_classification,
                                                                                                                                sort_events_randomly       = False,
                                                                                                                                include_smaller_last_batch = True
                                                                                                                                ):
                    batch_prediction, batch_loss = sess.run([predictions, loss], {input_data    : batch_data,
                                                                                  labels        : batch_labels,
                                                                                  event_weights : batch_event_weights})
 
                    training_batch_predictions.append(batch_prediction)
                    training_batch_losses.append(batch_loss)
 

                training_predictions                    = np.concatenate(training_batch_predictions, axis=0)
                training_labels, training_event_weights = training_data_set.get_labels_event_weights()
                training_accuracies.append(self._get_accuracy(training_labels, training_predictions, training_event_weights, network_type))

                training_losses.append(np.mean(training_batch_losses))


                #----------------------------------------------------------------------------------------------------
                # Make predictions for validation data set.

                validation_batch_predictions = list()

                for batch_data, batch_labels, batch_event_weights in validation_data_set.get_data_labels_event_weights_as_batches(batch_size                 = batch_size_classification,
                                                                                                                                  sort_events_randomly       = False,
                                                                                                                                  include_smaller_last_batch = True
                                                                                                                                  ):
                    batch_prediction = sess.run(predictions, {input_data    : batch_data,
                                                              labels        : batch_labels,
                                                              event_weights : batch_event_weights})

                    validation_batch_predictions.append(batch_prediction)


                validation_predictions                      = np.concatenate(validation_batch_predictions, axis=0)
                validation_labels, validation_event_weights = validation_data_set.get_labels_event_weights()
                validation_accuracies.append(self._get_accuracy(validation_labels, validation_predictions, validation_event_weights, network_type))


                #----------------------------------------------------------------------------------------------------

                print('{:^25} | {:^25.4e} | {:^25.4f} | {:^25.4f}'.format(epoch, training_losses[-1], training_accuracies[-1], validation_accuracies[-1]))


                epoch_end = time.time()
                epoch_durations.append(epoch_end - epoch_start)


                #----------------------------------------------------------------------------------------------------
                # Early stopping.

                if validation_accuracies[-1] > early_stopping['validation_accuracy']:
                    saver.save(sess, path_to_model_file)
 
                    early_stopping['validation_accuracy'] = validation_accuracies[-1]
                    early_stopping['epoch']               = epoch
 
 
                elif (epoch - early_stopping['epoch']) >= early_stopping_intervall:
                    print(110*'-')
                    print('Validation AUC has not increased for {} epochs. Achieved best validation auc score of {:.4f} in epoch {}'.format(early_stopping_intervall, early_stopping['validation_accuracy'], early_stopping['epoch']))
                    break


        #----------------------------------------------------------------------------------------------------

        network_and_training_properties_string = self._get_network_and_training_properties_string(network_type                 = network_type,
                                                                                                  number_of_input_neurons      = number_of_input_neurons,
                                                                                                  hidden_layers                = hidden_layers,
                                                                                                  activation_function_name     = activation_function_name,
                                                                                                  dropout_keep_probability     = dropout_keep_probability,
                                                                                                  l2_regularization_beta       = l2_regularization_beta,
                                                                                                  early_stopping_interval      = early_stopping_intervall,
                                                                                                  optimizer_options            = optimizer_options,
                                                                                                  batch_size_training          = batch_size_training,
                                                                                                  early_stopping               = early_stopping,
                                                                                                  mean_training_time_per_epoch = np.mean(epoch_durations)
                                                                                                  )

        with open(os.path.join(save_path, 'NN_Info.txt'), 'w') as NN_Info_output_file:
            NN_Info_output_file.write(network_and_training_properties_string)

        print(110*'-')
        print(network_and_training_properties_string, end='')
        print(110*'-')


        print('\n' + '========')
        print(       'FINISHED')
        print(       '========' + '\n')




    def _get_accuracy(self,
                      labels,
                      predictions,
                      event_weights,
                      network_type
                      ):
 

        if network_type == 'binary':
            accuracy = roc_auc_score(y_true = labels, y_score = predictions, sample_weight = event_weights)

        elif network_type == 'one-hot':
            array_true_prediction = np.zeros((labels.shape[1], labels.shape[1]), dtype=np.float32)
            
            index_true        = np.argmax(labels, axis=1)
            index_predictions = np.argmax(predictions, axis=1)

            for i in range(index_true.shape[0]):
                array_true_prediction[index_true[i]][index_predictions[i]] += event_weights[i]
     
            accuracy = np.diagonal(array_true_prediction).sum() / array_true_prediction.sum()
     
     
        return accuracy




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

        for i in range(1, len(weights)-1):
            layers.append( tf.nn.dropout(activation_function(tf.add(tf.matmul(layers[i-1], weights[i]), biases[i])), dropout_keep_probability) )

        logit = tf.add(tf.matmul(layers[-1], weights[-1]), biases[-1])


        return logit




    def _get_network_and_training_properties_string(self,
                                                    network_type,
                                                    number_of_input_neurons,
                                                    hidden_layers,
                                                    activation_function_name,
                                                    dropout_keep_probability,
                                                    l2_regularization_beta,
                                                    early_stopping_interval,
                                                    optimizer_options,
                                                    batch_size_training,
                                                    early_stopping,
                                                    mean_training_time_per_epoch
                                                    ):
        

        column_width = 55


        network_and_training_properties = str()


        network_and_training_properties += '{:{width}} {}\n'.format('Date:', datetime.datetime.now().strftime("%Y-%m-%d"), width=column_width)
        network_and_training_properties += '{:{width}} {}\n'.format('Time:', datetime.datetime.now().strftime("%H:%M:%S"), width=column_width)
        network_and_training_properties += '\n'


        network_and_training_properties += '{:{width}} {}\n'.format('Network type:', network_type, width=column_width)
        network_and_training_properties += '\n'


        network_and_training_properties += '{:{width}} {}\n'.format('Number of input variables:', number_of_input_neurons, width=column_width)
        network_and_training_properties += '\n'


        network_and_training_properties += '{:{width}} {}\n'.format('Hidden layers:',       hidden_layers,            width=column_width)
        network_and_training_properties += '{:{width}} {}\n'.format('Activation function:', activation_function_name, width=column_width)
        network_and_training_properties += '\n'


        network_and_training_properties += '{:{width}} {}\n'.format('Keep probability (dropout):', dropout_keep_probability, width=column_width)
        network_and_training_properties += '{:{width}} {}\n'.format('L2 regularization:',          l2_regularization_beta,   width=column_width)
        network_and_training_properties += '{:{width}} {}\n'.format('Early stopping interval:',    early_stopping_interval,  width=column_width)
        network_and_training_properties += '\n'


        network_and_training_properties += '{:{width}} {}\n'.format('Optimizer:', optimizer_options['optimizer_name'], width=column_width)

        optimizer_options_keys     = sorted([key for key in optimizer_options.keys() if 'learning_rate' not in key and key!='optimizer_name'])
        learning_rate_options_keys = sorted([key for key in optimizer_options.keys() if 'learning_rate' in key])

        for key in optimizer_options_keys:
            network_and_training_properties += '{:{width}} {}\n'.format(key+':', optimizer_options[key], width=column_width)

        for key in learning_rate_options_keys:
            network_and_training_properties += '{:{width}} {}\n'.format(key+':', optimizer_options[key], width=column_width)

        network_and_training_properties += '{:{width}} {}\n'.format('Batch size:', batch_size_training, width=column_width)
        network_and_training_properties += '\n'


        network_and_training_properties += '{:{width}} {}\n'.format('Early stopping epoch:', early_stopping['epoch'], width=column_width)

        if network_type == 'binary':
            network_and_training_properties += '{:{width}} {:.4f}\n'.format('ROC AUC (validation data set, early stopping epoch):', early_stopping['validation_accuracy'], width=column_width)

        elif network_type == 'one-hot':
            network_and_training_properties += '{:{width}} {:.4f}\n'.format('Accuracy (validation data set, early stopping epoch):', early_stopping['validation_accuracy'], width=column_width)

        network_and_training_properties += '\n'


        network_and_training_properties += '{:{width}} {:.2f} s\n'.format('Mean training time per epoch:', mean_training_time_per_epoch, width=column_width)


        return network_and_training_properties




    def _get_optimizer_global_step(self,
                                   optimizer_options,
                                   ):


        global_step = tf.Variable(0, trainable=False)

        if optimizer_options['learning_rate_decay']:
            initial_learning_rate = optimizer_options['learning_rate_decay_initial_value']

            decay_rate  = optimizer_options['learning_rate_decay_rate']
            decay_steps = optimizer_options['learning_rate_decay_steps']

            learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_rate=decay_rate, decay_steps=decay_steps)

        else:
            learning_rate = optimizer_options['learning_rate']


        if optimizer_options['optimizer_name'] == "Adam":
            beta1   = optimizer_options['beta1']
            beta2   = optimizer_options['beta2']
            epsilon = optimizer_options['epsilon']

            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)


        elif optimizer_options['optimizer_name'] == 'Adadelta':
            rho     = optimizer_options['rho']
            epsilon = optimizer_options['epsilon']

            optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=rho, epsilon=epsilon)


        elif optimizer_options['optimizer_name'] == 'Adagrad':
            initial_accumulator_value = optimizer_options['initial_accumulator_value']

            optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=initial_accumulator_value)


        elif optimizer_options['optimizer_name'] == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)


        elif optimizer_options['optimizer_name'] == 'Momentum':
            momentum     = optimizer_options['momentum']
            use_nesterov = optimizer_options['use_nesterov']

            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, use_nesterov=use_nesterov)


        else:
            sys.exit('Optimizer named "{}" is not implemented.'.format(optimizer_options['optimizer_name']))


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
