from __future__ import absolute_import, division, print_function

import os
import sys
import time
import itertools

import numpy as np
import tensorflow as tf

from mlp.mlp import MLP


class OneHotMLP(MLP):


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
              training_data,
              validation_data,
              batch_size,
              optimizer_options,
              gpu_usage
              ):


        print('\n' + 'TRAINING ONEHOT MLP' + '\n')


        if not os.path.isdir(save_path):
            sys.exit("Directory '" + save_path + "' doesn't exist." + "\n")


        path_to_model_file = os.path.join(save_path, '/{}.ckpt'.format(model_name))


        graph = tf.Graph()
        with graph.as_default():
            input_data    = tf.placeholder(tf.float32, [None, number_of_input_neurons], name='input')
            event_labels  = tf.placeholder(tf.float32, [None, number_of_output_neurons])
            event_weights = tf.placeholder(tf.float32, [None])

            feature_scaling_mean = tf.Variable(np.mean(train_data.x, axis=0).astype(np.float32), trainable=False,  name='feature_scaling_mean')
            feature_scaling_std  = tf.Variable(np.std(train_data.x, axis=0).astype(np.float32), trainable=False,  name='feature_scaling_std')
            input_data_scaled    = tf.div(tf.subtract(input_data, feature_scaling_mean), feature_scaling_std)

            weights, biases = self._get_initial_weights_biases(number_of_input_neurons, number_of_output_neurons, hidden_layers)

            logits      =               self._get_model(input_data_scaled, weights, biases, activation_function_name, dropout_keep_probability=dropout_keep_probability)
            predictions = tf.nn.softmax(self._get_model(input_data_scaled, weights, biases, activation_function_name, dropout_keep_probability=1), name='output')
            
            cross_entropy     = tf.nn.softmax_cross_entropy_with_logits(labels=event_labels, logits=logits)
            l2_regularization = beta * tf.add_n([tf.nn.l2_loss(w) for w in weights])
            loss              = tf.add(tf.reduce_mean(tf.multiply(event_weights, cross_entropy)), l2_regularization)
            
            optimizer, global_step = self._get_optimizer(optimizer_options)
            train_step             = optimizer.minimize(loss, global_step=global_step)

            saver = tf.train.Saver(weights + biases + [feature_scaling_mean, feature_scaling_std])
        
        
        config = self._get_session_config(gpu_usage)
        with tf.Session(config=config, graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            train_accuracy = []
            val_accuracy = []
            train_auc = []
            val_auc = []
            train_losses = []
            train_cats = []
            val_cats = []
            early_stopping = {'val_acc': -1.0, 'epoch': 0}

            print(100*'-')
            print('{:^25} | {:^25} | {:^25} | {:^25}'.format('Epoch', 'Training Loss', 'Training Accuracy', 'Validation Accuracy'))
            print(100*'-')

            cross_train_list = []
            cross_val_list = []
            weights_list = []
            
            train_start = time.time()
            for epoch in itertools.count(start=1, step=1):
                if (self.batch_decay == 'yes'):
                    batch_size = int(batch_size * (self.batch_decay_rate ** (1.0 /
                        self.batch_decay_steps)))
                # print(batch_size)
                total_batches = int(train_data.n/batch_size)
                epoch_loss = 0
                for _ in range(total_batches):
                    # train in batches
                    train_x, train_y, train_w=train_data.next_batch(batch_size)
                    _, train_loss, weights_for_plot, yps = sess.run([train_step,
                        loss, weights, y_], {x:train_x, y:train_y, w:train_w})
                    epoch_loss += train_loss
                weights_list.append(weights)
                train_losses.append(np.mean(epoch_loss))
                train_data.shuffle()

                # monitor training
                train_pre = sess.run(yy_, {x:train_data.x})
                train_corr, train_mistag, train_cross, train_cat = self._validate_epoch( 
                        train_pre, train_data.y, train_data.w)
                train_accuracy.append(train_corr / (train_corr + train_mistag))
                
                val_pre = sess.run(yy_, {x:val_data.x})
                val_corr, val_mistag, val_cross, val_cat = self._validate_epoch(val_pre,
                        val_data.y, val_data.w)
                val_accuracy.append(val_corr / (val_corr + val_mistag))
                
                
                print('{:^25} | {:^25.4e} | {:^25.4f} | {:^25.4f}'.format(epoch, train_losses[-1], train_accuracy[-1], val_accuracy[-1]))
                saver.save(sess, self.model_loc)
                cross_train_list.append(train_cross)
                cross_val_list.append(val_cross)
                train_cats.append(train_cat)
                val_cats.append(val_cat)


                # Check for early stopping.
                if (val_accuracy[-1] > early_stopping['val_acc']):
                    save_path = saver.save(sess, self.model_loc)
                    best_train_pred = train_pre
                    best_train_true = train_data.y
                    best_val_pred = val_pre
                    best_val_true = val_data.y
                    early_stopping['val_acc'] = val_accuracy[-1]
                    early_stopping['epoch'] = epoch
                elif ((epoch - early_stopping['epoch']) >= self.early_stop):
                    print(100*'-')
                    print('Early stopping invoked. Achieved best validation score of {:.4f} in epoch {}.'.format(early_stopping['val_acc'], early_stopping['epoch']))
                    break


            print(100*'-')
            train_end=time.time()
            dtime = train_end - train_start

            self._write_parameters(epochs, batch_size, keep_prob, beta,
                    dtime, early_stopping, val_accuracy[-1])





    def _get_accuracy(self,
                      prediction,
                      labels,
                      weights
                      ):


        array_true_prediction = np.zeros((labels.shape[1], labels.shape[1]), dtype=np.float32)
        index_true = np.argmax(labels, axis=1)
        index_pred = np.argmax(pred, axis=1)
        for i in range(index_true.shape[0]):
            array_true_prediction[index_true[i]][index_pred[i]] += weights[i]

        accuracy = np.diagonal(array_true_prediction).sum() / array_true_prediction.sum()


        return accuracy
