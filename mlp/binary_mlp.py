from __future__ import absolute_import, division, print_function

import time

import numpy as np

import tensorflow as tf

from sklearn.metrics import roc_auc_score

from mlp.mlp import MLP


class BinaryMLP(MLP):

    def train(self, train_data, val_data, epochs, batch_size,
              lr, optimizer, early_stop, keep_prob, beta, gpu_usage,
              momentum=None, lr_decay=None):
        
        
        
        self._lr = lr
        self._optimizer = optimizer
        self._momentum = momentum
        self._lr_decay = lr_decay
        
        train_graph = tf.Graph()
        with train_graph.as_default():
            x = tf.placeholder(tf.float32, [None, self._n_variables], name='input')
            y = tf.placeholder(tf.float32, [None, 1])
            w = tf.placeholder(tf.float32, [None, 1])

            x_mean = tf.Variable(np.mean(train_data.x, axis=0).astype(np.float32),
                                 trainable=False,  name='x_mean')
            x_std = tf.Variable(np.std(train_data.x, axis=0).astype(np.float32),
                                trainable=False,  name='x_std')
            x_scaled = tf.div(tf.subtract(x, x_mean), x_std, name='x_scaled')

            weights, biases = self._get_parameters()

            # prediction, y_ is used for training, yy_ used for makin new
            # predictions
            y_ = self._model(x_scaled, weights, biases, keep_prob)
            yy_ = tf.nn.sigmoid(self._model(x_scaled, weights, biases), name='output')

            # loss function
            xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, labels=y)
            l2_regularization = beta * tf.add_n([tf.nn.l2_loss(w) for w in weights])
            loss = tf.add(tf.reduce_mean(tf.multiply(w, xentropy)), l2_regularization,
                                  name='loss')

            # optimizer
            opt, global_step = self._get_optimizer()
            train_step = opt.minimize(loss, global_step=global_step)

            # initialize the variables
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(weights + biases + [x_mean, x_std])

        

        config = self._get_session_config(gpu_usage)
        with tf.Session(config=config, graph=train_graph) as sess:
            self.model_loc = self._savedir + '/{}.ckpt'.format(self._name)
            sess.run(init)
            
            train_auc = []
            val_auc = []
            train_loss = []
            early_stopping  = {'auc': 0.0, 'epoch': 0}
            epoch_durations = []
            print(90*'-')
            print('Train model: {}'.format(self.model_loc))
            print(90*'-')
            print('{:^20} | {:^20} | {:^20} |{:^25}'.format(
                'Epoch', 'Training Loss','AUC Training Score',
                'AUC Validation Score'))
            print(90*'-')

            for epoch in range(1, epochs+1):
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
                val_pre = sess.run(yy_, {x : val_data.x})
                val_auc.append(roc_auc_score(y_true = val_data.y, y_score = val_pre, sample_weight = val_data.w))
                
                print('{:^20} | {:^20.4e} | {:^20.4f} | {:^25.4f}'
                      .format(epoch, train_loss[-1],
                              train_auc[-1], val_auc[-1]))

                if early_stop:
                    # check for early stopping, only save model if val_auc has
                    # increased
                    if val_auc[-1] > early_stopping['auc']:
                        save_path = saver.save(sess, self.model_loc)
                        early_stopping['auc'] = val_auc[-1]
                        early_stopping['epoch'] = epoch
                        early_stopping['val_pre'] = val_pre
                    elif (epoch - early_stopping['epoch']) >= early_stop:
                        print(125*'-')
                        print('Validation AUC has not increased for {} epochs. ' \
                              'Achieved best validation auc score of {:.4f} ' \
                              'in epoch {}'.format(early_stop, 
                                  early_stopping['auc'],
                                  early_stopping['epoch']))
                        break
                else:
                    save_path = saver.save(sess, self.model_loc)
                
                # set internal dataframe index to 0
                
                epoch_end = time.time()
                epoch_durations.append(epoch_end - epoch_start)

            print(90*'-')
            self._write_parameters(batch_size, keep_prob, beta,
                                   np.mean(epoch_durations), early_stopping)
            print('Model saved in: {}'.format(save_path))
            print(90*'-')
