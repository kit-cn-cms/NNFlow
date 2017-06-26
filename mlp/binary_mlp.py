from __future__ import absolute_import, division, print_function

import os
import sys
import time
import itertools

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from mlp.mlp import MLP
from dataframe.dataframe import DataFrame


class BinaryMLP(MLP):

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
