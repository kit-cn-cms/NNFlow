from __future__ import absolute_import, division, print_function

import os
import sys
import time
import datetime
import itertools

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from NNFlow.data_frame.data_frame import DataFrame




class NeuralNetworkTrainer(object):


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
              optimizer,
              batch_size_training,
              batch_size_classification,
              session_config
              ):


        print('\n' + '=======================')
        print(       'TRAINING NEURAL NETWORK')
        print(       '=======================' + '\n')

 
        if not os.path.isdir(save_path):
            sys.exit("Directory '" + save_path + "' doesn't exist." + "\n")
 
        if not os.path.isfile(path_to_training_data_set):
            sys.exit("File '" + path_to_training_data_set + "' doesn't exist." + "\n")
 
        if not os.path.isfile(path_to_validation_data_set):
            sys.exit("File '" + path_to_validation_data_set + "' doesn't exist." + "\n")
 
 
        path_to_model_file         = os.path.join(save_path, '{}.ckpt'.format(model_name))
        directory_model_properties = os.path.join(save_path, 'model_properties')

        if not os.path.isdir(directory_model_properties):
            os.mkdir(directory_model_properties)


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
 
            tf_optimizer, global_step = optimizer.get_optimizer_global_step()
            train_step                = tf_optimizer.minimize(loss, global_step=global_step)
 
            saver = tf.train.Saver(weights + biases + [feature_scaling_mean, feature_scaling_std])
 

        #----------------------------------------------------------------------------------------------------

        config = session_config.get_config()
        with tf.Session(config=config, graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
 
 
            training_accuracies   = list()
            validation_accuracies = list()
            training_losses       = list()
            early_stopping        = {'validation_accuracy': -1.0, 'epoch': 0}
            epoch_durations       = list()
            training_development  = str()


            output_separator = 110*'-' + '\n'
            if network_type == 'binary':
                training_development_heading = '{:^25} | {:^25} | {:^25} | {:^25}\n'.format('Epoch', 'Training data: loss', 'Training data: ROC AUC', 'Validation data: ROC AUC')
            elif network_type == 'one-hot':
                training_development_heading = '{:^25} | {:^25} | {:^25} | {:^25}\n'.format('Epoch', 'Training data: loss', 'Training data: accuracy', 'Validation data: accuracy')

            print('\n',                         end='')
            print(output_separator,             end='')
            print(training_development_heading, end='')
            print(output_separator,             end='')

            training_development += output_separator
            training_development += training_development_heading
            training_development += output_separator


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

                training_batch_predictions_list = list()
                training_batch_loss_list        = list()

                for batch_data, batch_labels, batch_event_weights in training_data_set.get_data_labels_event_weights_as_batches(batch_size                 = batch_size_classification,
                                                                                                                                sort_events_randomly       = False,
                                                                                                                                include_smaller_last_batch = True
                                                                                                                                ):
                    batch_predictions, batch_loss = sess.run([predictions, loss], {input_data    : batch_data,
                                                                                   labels        : batch_labels,
                                                                                   event_weights : batch_event_weights})
 
                    training_batch_predictions_list.append(batch_predictions)
                    training_batch_loss_list.append(batch_loss)
 

                training_predictions                    = np.concatenate(training_batch_predictions_list, axis=0)
                training_labels, training_event_weights = training_data_set.get_labels_event_weights()
                training_accuracies.append(self._get_accuracy(training_labels, training_predictions, training_event_weights, network_type))

                training_losses.append(np.mean(training_batch_loss_list))


                #----------------------------------------------------------------------------------------------------
                # Make predictions for validation data set.

                validation_batch_predictions_list = list()

                for batch_data, batch_labels, batch_event_weights in validation_data_set.get_data_labels_event_weights_as_batches(batch_size                 = batch_size_classification,
                                                                                                                                  sort_events_randomly       = False,
                                                                                                                                  include_smaller_last_batch = True
                                                                                                                                  ):
                    batch_predictions = sess.run(predictions, {input_data : batch_data})

                    validation_batch_predictions_list.append(batch_predictions)


                validation_predictions                      = np.concatenate(validation_batch_predictions_list, axis=0)
                validation_labels, validation_event_weights = validation_data_set.get_labels_event_weights()
                validation_accuracies.append(self._get_accuracy(validation_labels, validation_predictions, validation_event_weights, network_type))


                #----------------------------------------------------------------------------------------------------

                training_development_epoch = '{:^25} | {:^25.4e} | {:^25.4f} | {:^25.4f}\n'.format(epoch, training_losses[-1], training_accuracies[-1], validation_accuracies[-1])
                print(training_development_epoch, end='')
                training_development += training_development_epoch


                #----------------------------------------------------------------------------------------------------

                epoch_end = time.time()
                epoch_durations.append(epoch_end - epoch_start)


                #----------------------------------------------------------------------------------------------------
                # Early stopping.

                if validation_accuracies[-1] > early_stopping['validation_accuracy']:
                    saver.save(sess, path_to_model_file)
 
                    early_stopping['validation_accuracy'] = validation_accuracies[-1]
                    early_stopping['epoch']               = epoch
 
 
                elif (epoch - early_stopping['epoch']) >= early_stopping_intervall:
                    if network_type == 'binary':
                        training_development_early_stop = 'ROC AUC on validation data has not increased for {} epochs. Achieved best ROC AUC of {:.4f} in epoch {}.\n'.format(early_stopping_intervall, early_stopping['validation_accuracy'], early_stopping['epoch'])
                    elif network_type == 'one-hot':
                        training_development_early_stop = 'Accuracy on validation data has not increased for {} epochs. Achieved best accuracy of {:.4f} in epoch {}.\n'.format(early_stopping_intervall, early_stopping['validation_accuracy'], early_stopping['epoch'])
                    
                    print(output_separator,                end='')
                    print(training_development_early_stop, end='')
                    print(output_separator,                end='')

                    training_development += output_separator
                    training_development += training_development_early_stop
                    training_development += output_separator

                    break


        #----------------------------------------------------------------------------------------------------

        network_and_training_properties_string = self._get_network_and_training_properties_string(network_type                 = network_type,
                                                                                                  number_of_input_neurons      = number_of_input_neurons,
                                                                                                  hidden_layers                = hidden_layers,
                                                                                                  activation_function_name     = activation_function_name,
                                                                                                  dropout_keep_probability     = dropout_keep_probability,
                                                                                                  l2_regularization_beta       = l2_regularization_beta,
                                                                                                  early_stopping_interval      = early_stopping_intervall,
                                                                                                  optimizer                    = optimizer,
                                                                                                  batch_size_training          = batch_size_training,
                                                                                                  early_stopping               = early_stopping,
                                                                                                  mean_training_time_per_epoch = np.mean(epoch_durations)
                                                                                                  )

        with open(os.path.join(directory_model_properties, 'NN_Info.txt'), 'w') as NN_Info_output_file:
            NN_Info_output_file.write(network_and_training_properties_string)

        print(network_and_training_properties_string, end='')
        print(output_separator,                       end='')


        with open(os.path.join(directory_model_properties, 'training_development.txt'), 'w') as training_development_output_file:
            training_development_output_file.write(training_development)

        self._plot_training_development(directory_model_properties, network_type, training_accuracies, validation_accuracies, early_stopping)


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
                                                    optimizer,
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


        optimizer_options = optimizer.get_optimizer_properties()

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




    def _plot_training_development(self,
                                   directory_model_properties,
                                   network_type,
                                   training_accuracies,
                                   validation_accuracies,
                                   early_stopping,
                                   ):


        plt.clf()


        plt.plot(range(1, len(training_accuracies)  +1), training_accuracies,   color='#1f77b4', label='Training',   ls='', marker='o')
        plt.plot(range(1, len(validation_accuracies)+1), validation_accuracies, color='#ff7f0e', label='Validation', ls='', marker='o')

        plt.axvline(x=early_stopping['epoch'], color='r')


        plt.xlabel('Epoch')
        if network_type == 'binary':
            plt.ylabel('ROC AUC')
        elif network_type == 'one-hot':
            plt.ylabel('Accuracy')

        plt.legend(loc='best', frameon=False)


        plt.savefig(os.path.join(directory_model_properties, 'training_development.pdf'))


        plt.clf()
