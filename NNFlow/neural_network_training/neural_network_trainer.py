from __future__ import absolute_import, division, print_function

import os
import sys
import time
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from NNFlow.data_frame.data_frame                             import DataFrame              as NNFlowDataFrame
from NNFlow.softmax_output_processor.softmax_output_processor import SoftmaxOutputProcessor as NNFlowSoftmaxOutputProcessor




class NeuralNetworkTrainer(object):


    def __init__(self):


        self._softmax_output_processor = NNFlowSoftmaxOutputProcessor()




    def train(self,
              save_path,
              model_name,
              model_id,
              network_type,
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
        directory_plots            = os.path.join(save_path, 'plots')

        if not os.path.isdir(directory_model_properties):
            os.mkdir(directory_model_properties)
        if not os.path.isdir(directory_plots):
            os.mkdir(directory_plots)


        #----------------------------------------------------------------------------------------------------
        # Load data.
 
        training_data_set   = NNFlowDataFrame(path_to_training_data_set)
        validation_data_set = NNFlowDataFrame(path_to_validation_data_set)

        number_of_input_neurons  = training_data_set.get_number_of_input_neurons()
        number_of_output_neurons = training_data_set.get_number_of_output_neurons()


        #----------------------------------------------------------------------------------------------------
        # Graph.

        tf_graph = tf.Graph()
        with tf_graph.as_default():
            tf_input_data    = tf.placeholder(tf.float32, [None, number_of_input_neurons], name='input')
            tf_event_weights = tf.placeholder(tf.float32, [None])
 
            tf_feature_scaling_mean = tf.Variable(np.mean(training_data_set.get_data(), axis=0).astype(np.float32), trainable=False,  name='feature_scaling_mean')
            tf_feature_scaling_std  = tf.Variable(np.std(training_data_set.get_data(), axis=0).astype(np.float32), trainable=False,  name='feature_scaling_std')
            tf_input_data_scaled    = tf.div(tf.subtract(tf_input_data, tf_feature_scaling_mean), tf_feature_scaling_std)
 
            tf_weights, tf_biases = self._get_initial_tf_weights_tf_biases(number_of_input_neurons, number_of_output_neurons, hidden_layers)


            if network_type == 'binary':
                tf_labels = tf.placeholder(tf.float32, [None])

                tf_logits         =               tf.reshape(self._get_tf_logits(tf_input_data_scaled,tf_weights,tf_biases,activation_function_name,dropout_keep_probability),[-1])
                tf_network_output = tf.nn.sigmoid(tf.reshape(self._get_tf_logits(tf_input_data_scaled,tf_weights,tf_biases,activation_function_name,dropout_keep_probability=1), [-1]), name='output')

                tf_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf_logits, labels=tf_labels)


            elif network_type == 'multiclass':
                tf_labels = tf.placeholder(tf.float32, [None, number_of_output_neurons])

                tf_logits         =               self._get_tf_logits(tf_input_data_scaled, tf_weights, tf_biases, activation_function_name, dropout_keep_probability=dropout_keep_probability)
                tf_network_output = tf.nn.softmax(self._get_tf_logits(tf_input_data_scaled, tf_weights, tf_biases, activation_function_name, dropout_keep_probability=1), name='output')

                tf_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=tf_logits)


            tf_l2_regularization = l2_regularization_beta * tf.add_n([tf.nn.l2_loss(w) for w in tf_weights])
            tf_loss              = tf.add(tf.reduce_mean(tf.multiply(tf_event_weights, tf_cross_entropy)), tf_l2_regularization)
 
            tf_optimizer, tf_global_step = optimizer.get_tf_optimizer_tf_global_step()
            tf_training_step             = tf_optimizer.minimize(tf_loss, global_step=tf_global_step)


            tf_model_id        = tf.Variable(model_id,                                trainable=False, name='model_id')
            tf_input_variables = tf.Variable(training_data_set.get_input_variables(), trainable=False, name='input_variables')
            tf_output_labels   = tf.Variable(training_data_set.get_output_labels(),   trainable=False, name='output_labels')
            tf_network_type    = tf.Variable(training_data_set.get_network_type(),    trainable=False, name='network_type')
            tf_preselection    = tf.Variable(training_data_set.get_preselection(),    trainable=False, name='preselection')


            tf_saver = tf.train.Saver(tf_weights + tf_biases + [tf_feature_scaling_mean, tf_feature_scaling_std, tf_model_id, tf_input_variables, tf_output_labels, tf_network_type, tf_preselection])
 

        #----------------------------------------------------------------------------------------------------

        tf_config = session_config.get_tf_config()
        with tf.Session(config=tf_config, graph=tf_graph) as tf_session:
            tf_session.run(tf.global_variables_initializer())
 
 
            training_roc_auc      = list()
            validation_roc_auc    = list()
            training_losses       = list()
            early_stopping        = {'validation_roc_auc': -1.0, 'epoch': 0}
            epoch_durations       = list()
            training_development  = str()


            output_separator = 130*'-' + '\n'
            if network_type == 'binary':
                training_development_heading = '{:^30} | {:^30} | {:^30} | {:^30}\n'.format('Epoch', 'Training Data: Loss', 'Training Data: ROC AUC', 'Validation Data: ROC AUC')
            elif network_type == 'multiclass':
                training_development_heading = '{:^30} | {:^30} | {:^30} | {:^30}\n'.format('Epoch', 'Training Data: Loss', 'Training Data: Mean ROC AUC', 'Validation Data: Mean ROC AUC')

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
                    tf_session.run(tf_training_step, {tf_input_data    : batch_data,
                                                      tf_labels        : batch_labels,
                                                      tf_event_weights : batch_event_weights})


                #----------------------------------------------------------------------------------------------------
                # Calculate network_output for training data set.

                training_batch_network_output_list = list()
                training_batch_loss_list           = list()

                for batch_data, batch_labels, batch_event_weights in training_data_set.get_data_labels_event_weights_as_batches(batch_size                 = batch_size_classification,
                                                                                                                                sort_events_randomly       = False,
                                                                                                                                include_smaller_last_batch = True
                                                                                                                                ):
                    batch_network_output, batch_loss = tf_session.run([tf_network_output, tf_loss], {tf_input_data    : batch_data,
                                                                                                     tf_labels        : batch_labels,
                                                                                                     tf_event_weights : batch_event_weights})
 
                    training_batch_network_output_list.append(batch_network_output)
                    training_batch_loss_list.append(batch_loss)
 

                training_network_output                 = np.concatenate(training_batch_network_output_list, axis=0)
                training_labels, training_event_weights = training_data_set.get_labels_event_weights()
                training_roc_auc.append(self._get_roc_auc(training_labels, training_network_output, training_event_weights, network_type))

                training_losses.append(np.mean(training_batch_loss_list))


                #----------------------------------------------------------------------------------------------------
                # Calculate network_output for validation data set.

                validation_batch_network_output_list = list()

                for batch_data, batch_labels, batch_event_weights in validation_data_set.get_data_labels_event_weights_as_batches(batch_size                 = batch_size_classification,
                                                                                                                                  sort_events_randomly       = False,
                                                                                                                                  include_smaller_last_batch = True
                                                                                                                                  ):
                    batch_network_output = tf_session.run(tf_network_output, {tf_input_data : batch_data})

                    validation_batch_network_output_list.append(batch_network_output)


                validation_network_output                   = np.concatenate(validation_batch_network_output_list, axis=0)
                validation_labels, validation_event_weights = validation_data_set.get_labels_event_weights()
                validation_roc_auc.append(self._get_roc_auc(validation_labels, validation_network_output, validation_event_weights, network_type))


                #----------------------------------------------------------------------------------------------------

                training_development_epoch = '{:^30} | {:^30.4e} | {:^30.4f} | {:^30.4f}\n'.format(epoch, training_losses[-1], training_roc_auc[-1], validation_roc_auc[-1])
                print(training_development_epoch, end='')
                training_development += training_development_epoch


                #----------------------------------------------------------------------------------------------------

                epoch_end = time.time()
                epoch_durations.append(epoch_end - epoch_start)


                #----------------------------------------------------------------------------------------------------
                # Early stopping.

                if validation_roc_auc[-1] > early_stopping['validation_roc_auc']:
                    tf_saver.save(tf_session, path_to_model_file)
 
                    early_stopping['validation_roc_auc'] = validation_roc_auc[-1]
                    early_stopping['epoch']              = epoch
 
 
                elif (epoch - early_stopping['epoch']) >= early_stopping_intervall:
                    if network_type == 'binary':
                        training_development_early_stop = 'ROC AUC on validation data has not increased for {} epochs. Achieved best ROC AUC of {:.4f} in epoch {}.\n'.format(early_stopping_intervall, early_stopping['validation_roc_auc'], early_stopping['epoch'])
                    elif network_type == 'multiclass':
                        training_development_early_stop = 'Mean ROC AUC on validation data has not increased for {} epochs. Achieved best mean ROC AUC of {:.4f} in epoch {}.\n'.format(early_stopping_intervall, early_stopping['validation_roc_auc'], early_stopping['epoch'])
                    
                    print(output_separator,                end='')
                    print(training_development_early_stop, end='')
                    print(output_separator,                end='')

                    training_development += output_separator
                    training_development += training_development_early_stop
                    training_development += output_separator

                    break


        #----------------------------------------------------------------------------------------------------

        network_and_training_properties_string = self._get_network_and_training_properties_string(model_id                     = model_id,
                                                                                                  network_type                 = network_type,
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

        self._plot_training_development(directory_plots, network_type, training_roc_auc, validation_roc_auc, early_stopping)


        with open(os.path.join(directory_model_properties, 'inputVariables.txt'), 'w') as outputfile_input_variables:
            for variable in training_data_set.get_input_variables():
                outputfile_input_variables.write(variable + '\n')


        with open(os.path.join(directory_model_properties, 'outputLabels.txt'), 'w') as outputfile_output_labels:
            for output_label in training_data_set.get_output_labels()
                outputfile_output_labels.write(output_label + '\n')


        with open(os.path.join(directory_model_properties, 'preselection.txt'), 'w') as outputfile_preselection:
            outputfile_preselection.write(training_data_set.get_preselection() + '\n')


        print('\n' + '========')
        print(       'FINISHED')
        print(       '========' + '\n')




    def _get_initial_tf_weights_tf_biases(self,
                                          number_of_input_neurons,
                                          number_of_output_neurons,
                                          hidden_layers
                                          ):


        tf_weights = [tf.Variable(tf.random_normal([number_of_input_neurons, hidden_layers[0]], stddev=tf.sqrt(2.0/number_of_input_neurons)), name = 'W_1')]
        tf_biases  = [tf.Variable(tf.fill(dims=[hidden_layers[0]], value=0.1), name = 'B_1')]


        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                tf_weights.append(tf.Variable(tf.random_normal(shape = [hidden_layers[i-1], hidden_layers[i]], stddev = tf.sqrt(2.0/hidden_layers[i-1])), name='W_{}'.format(i+1)))
                tf_biases.append(tf.Variable(tf.fill(dims=[hidden_layers[i]], value = 0.1), name='B_{}'.format(i+1)))


        tf_weights.append(tf.Variable(tf.random_normal([hidden_layers[-1], number_of_output_neurons], stddev=tf.sqrt(2.0/hidden_layers[-1])), name = 'W_out'))
        tf_biases.append(tf.Variable(tf.fill(dims=[number_of_output_neurons], value=0.1), name = 'B_out'))


        return tf_weights, tf_biases




    def _get_network_and_training_properties_string(self,
                                                    model_id,
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
        

        column_width = 60


        network_and_training_properties = str()


        network_and_training_properties += '{:{width}} {}\n'.format('Model ID:', model_id, width=column_width)
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
            network_and_training_properties += '{:{width}} {:.4f}\n'.format('ROC AUC (validation data set, early stopping epoch):', early_stopping['validation_roc_auc'], width=column_width)

        elif network_type == 'multiclass':
            network_and_training_properties += '{:{width}} {:.4f}\n'.format('Mean ROC AUC (validation data set, early stopping epoch):', early_stopping['validation_roc_auc'], width=column_width)

        network_and_training_properties += '\n'


        network_and_training_properties += '{:{width}} {:.2f} s\n'.format('Mean training time per epoch:', mean_training_time_per_epoch, width=column_width)


        return network_and_training_properties




    def _get_roc_auc(self,
                     labels,
                     network_output,
                     event_weights,
                     network_type
                     ):
 

        if network_type == 'binary':
            roc_auc = roc_auc_score(y_true = labels, y_score = network_output, sample_weight = event_weights)

        elif network_type == 'multiclass':
            roc_auc = self._softmax_output_processor.get_mean_roc_auc(labels, network_output, event_weights)
     
     
        return roc_auc




    def _get_tf_activation_function(self,
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




    def _get_tf_logits(self,
                       tf_data,
                       tf_weights,
                       tf_biases,
                       activation_function_name,
                       dropout_keep_probability
                       ):


        tf_activation_function = self._get_tf_activation_function(activation_function_name)

        tf_layers = list()
        tf_layers.append( tf_activation_function(tf.add(tf.matmul(tf_data, tf_weights[0]), tf_biases[0])) )

        for i in range(1, len(tf_weights)-1):
            tf_layers.append( tf.nn.dropout(tf_activation_function(tf.add(tf.matmul(tf_layers[i-1], tf_weights[i]), tf_biases[i])), dropout_keep_probability) )

        tf_logits = tf.add(tf.matmul(tf_layers[-1], tf_weights[-1]), tf_biases[-1])


        return tf_logits




    def _plot_training_development(self,
                                   directory_plots,
                                   network_type,
                                   training_roc_auc,
                                   validation_roc_auc,
                                   early_stopping,
                                   ):


        plt.clf()


        plt.plot(range(1, len(training_roc_auc)  +1), training_roc_auc,   color='#1f77b4', label='Training',   ls='', marker='o')
        plt.plot(range(1, len(validation_roc_auc)+1), validation_roc_auc, color='#ff7f0e', label='Validation', ls='', marker='o')

        plt.axvline(x=early_stopping['epoch'], color='r')


        plt.xlabel('Epoch')
        if network_type == 'binary':
            plt.ylabel('ROC AUC')
        elif network_type == 'multiclass':
            plt.ylabel('Mean ROC AUC')

        plt.legend(loc='best', frameon=False)


        plt.savefig(os.path.join(directory_plots, 'training_development.pdf'))


        plt.clf()
