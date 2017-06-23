from __future__ import absolute_import, division, print_function

import os
import time
import itertools

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from mlp.mlp import MLP


class BinaryMLP(MLP):

    def train(self,
              save_path,
              model_name,
              number_of_input_neurons,
              hidden_layers,
              activation_function_name,
              dropout_keep_probability,
              l2_regularization_beta,
              early_stopping_intervall,
              training_data,
              validation_data,
              batch_size,
              optimizer_options,
              gpu_usage
              ):


        path_to_model_file = os.path.join(save_path, '/{}.ckpt'.format(model_name))
 

        graph = tf.Graph()
        with graph.as_default():
            input_data    = tf.placeholder(tf.float32, [None, number_of_input_neurons], name='input')
            event_labels  = tf.placeholder(tf.float32, [None])
            event_weights = tf.placeholder(tf.float32, [None])

            feature_scaling_mean = tf.Variable(np.mean(training_data.x, axis=0).astype(np.float32), trainable=False,  name='feature_scaling_mean')
            feature_scaling_std  = tf.Variable(np.std(training_data.x, axis=0).astype(np.float32), trainable=False,  name='feature_scaling_std')
            input_data_scaled    = tf.div(tf.subtract(input_data, feature_scaling_mean), feature_scaling_std)

            weights, biases = self._get_initial_weights_biases(number_of_input_neurons, number_of_output_neurons, hidden_layers)

            logits      =               tf.reshape(self._get_model(x_scaled, weights, biases, activation_function_name, dropout_keep_probability=dropout_keep_probability), [-1])
            predictions = tf.nn.sigmoid(tf.reshape(self._get_model(x_scaled, weights, biases, activation_function_name, dropout_keep_probability=1                       ), [-1]), name='output')

            cross_entropy     = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=event_labels)
            l2_regularization = beta * tf.add_n([tf.nn.l2_loss(w) for w in weights])
            loss              = tf.add(tf.reduce_mean(tf.multiply(w, cross_entropy)), l2_regularization)

            optimizer, global_step = self._get_optimizer(optimizer_options)
            train_step = optimizer.minimize(loss, global_step=global_step)

            saver = tf.train.Saver(weights + biases + [feature_scaling_mean, feature_scaling_std])


        config = self._get_session_config(gpu_usage)
        with tf.Session(config=config, graph=graph) as sess:
            print('\n' + 'TRAINING BINARY MLP' + '\n')

            sess.run(tf.global_variables_initializer())


            training_roc_auc   = list()
            validation_roc_auc = list()
            training_loss      = list()
            early_stopping     = {'auc': 0.0, 'epoch': 0}
            epoch_durations    = list()


            print(100*'-')
            print('{:^25} | {:^25} | {:^25} |{:^25}'.format('Epoch', 'Training data: loss', 'Training data: ROC AUC', 'Validation data: ROC AUC'))
            print(100*'-')

            for epoch in itertools.count(start=0, step=1)
                epoch_start = time.time()
                total_batches = int(train_data.n/batch_size)
                epoch_loss = 0
                for _ in range(total_batches):
                    train_x, train_y, train_w= train_data.next_batch(batch_size) 
                    batch_loss, _  = sess.run([loss, train_step],
                                              {x: train_x,
                                               y: train_y,
                                               w: train_w})
                    epoch_loss += batch_loss
                    
                # monitor training
                train_data.shuffle()
                train_loss.append(epoch_loss/total_batches)
                train_pre = []
                for batch in range(0, total_batches):
                    train_x, _, _ = train_data.next_batch(batch_size)
                    pred = sess.run(yy_, {x: train_x})
                    train_pre.append(pred)
                train_pre = np.concatenate(train_pre, axis=0)
                train_auc.append(roc_auc_score(y_true = train_data.y[:total_batches*batch_size], y_score = train_pre, sample_weight = train_data.w[:total_batches*batch_size]))
                validation_prediction = sess.run(yy_, {input_data : val_data.x})
                validation_roc_auc.append(roc_auc_score(y_true = val_data.y, y_score = val_pre, sample_weight = val_data.w))
                
                print('{:^25} | {:^25.4e} | {:^25.4f} | {:^25.4f}'.format(epoch, train_loss[-1], train_auc[-1], val_auc[-1]))

                # check for early stopping, only save model if val_auc has increased
                if val_auc[-1] > early_stopping['auc']:
                    saver.save(sess, path_to_model_file)
                    early_stopping['auc'] = val_auc[-1]
                    early_stopping['epoch'] = epoch
                    early_stopping['val_pre'] = val_pre
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
