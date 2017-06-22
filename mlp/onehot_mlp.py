from __future__ import absolute_import, division, print_function

import time
import datetime

import numpy as np

import tensorflow as tf

from mlp.mlp import MLP


class OneHotMLP(MLP):

    def init_old(self, n_features, h_layers, out_size, savedir, labels_text,
            branchlist, act_func='tanh'):
        """Initializes the Classifier.

        Arguments:
        ----------------
        n_features (int):
            The number of input features.
        h_layers (list):
            A list representing the hidden layers. Each entry gives the number
            of neurons in the equivalent layer, [30,40,20] would describe a
            network of three hidden layers, containing 30, 40 and 20 neurons.
        out_size (int):
            The size of the one-hot output vector.

        Attributes:
        ----------------
        savedir (str):
            Path to the directory everything will be saved to.
        labels_text (list):
            List of strings containing the labels for the plots.
        branchlist (list):
            List of strings containing the branches used.
        sig_weight (float):
            Weight of ttH events.
        bg_weight (float):
            Weight of ttbar events.
        act_func (string):
            Activation function.
        """

        self.labels_text = labels_text
        self.branchlist = branchlist


        # create directory if necessary
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)
        
        self.cross_savedir = self.savedir + '/cross_checks'
        if not os.path.isdir(self.cross_savedir):
            os.makedirs(self.cross_savedir)
        
        self.hists_savedir_train = self.cross_savedir + '/hists_train/'
        if not os.path.isdir(self.hists_savedir_train):
            os.makedirs(self.hists_savedir_train)
        self.hists_savedir_val = self.cross_savedir + '/hists_val/'
        if not os.path.isdir(self.hists_savedir_val):
            os.makedirs(self.hists_savedir_val)
        self.weights_savedir = self.cross_savedir + '/weights/'
        if not os.path.isdir(self.weights_savedir):
            os.makedirs(self.weights_savedir)
        self.mistag_savedir = self.cross_savedir + '/mistag/'
        if not os.path.isdir(self.mistag_savedir):
            os.makedirs(self.mistag_savedir)




    def train(self, train_data, val_data, optimizer='Adam', epochs = 10, batch_size = 100,
            learning_rate = 1e-3, keep_prob = 0.9, beta = 0.0, out_size=6, 
            optimizer_options=[], enable_early='no', early_stop=10, 
            decay_learning_rate='no', dlrate_options=[], batch_decay='no', 
            batch_decay_options=[], gpu_usage=None):


        self.optname = optimizer
        self.learning_rate = learning_rate
        self.optimizer_options = optimizer_options
        self.enable_early = enable_early
        self.early_stop = early_stop
        self.decay_learning_rate = decay_learning_rate
        self.decay_learning_rate_options = dlrate_options
        self.batch_decay = batch_decay
        self.batch_decay_options = batch_decay_options

        if (self.batch_decay == 'yes'):
            try:
                self.batch_decay_rate = batch_decay_options[0]
            except IndexError:
                self.batch_decay_rate = 0.95
            try:
                self.batch_decay_steps = batch_decay_options[1]
            except IndexError:
                # Batch size decreases over 10 epochs
                self.batch_decay_steps = 10

        train_graph = tf.Graph()
        with train_graph.as_default():
            x = tf.placeholder(tf.float32, [None, self._number_of_input_neurons], name='input')
            y = tf.placeholder(tf.float32, [None, out_size])
            w = tf.placeholder(tf.float32, [None])

            x_mean = tf.Variable(np.mean(train_data.x, axis=0).astype(np.float32), trainable=False,  name='x_mean')
            x_std = tf.Variable(np.std(train_data.x, axis=0).astype(np.float32), trainable=False,  name='x_std')
            x_scaled = tf.div(tf.subtract(x, x_mean), x_std, name='x_scaled')

            weights, biases = self._get_parameters()

            # prediction
            y_ = self._model(x_scaled, weights, biases, keep_prob)
            # prediction for validation
            yy_ = tf.nn.softmax(self._model(x_scaled, weights, biases), name='output')
            # Cross entropy
            xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_)
            l2_regularization = beta * tf.add_n([tf.nn.l2_loss(w) for w in weights])
            loss = tf.add(tf.reduce_mean(tf.multiply(w, xentropy)), l2_regularization, 
                    name='loss')
            
            # optimizer
            optimizer, global_step = self._build_optimizer()
            train_step = optimizer.minimize(loss, global_step=global_step)

            # initialize all variables
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(weights + biases + [x_mean, x_std])
        
        
        config = self._get_session_config(gpu_usage)
        with tf.Session(config=config, graph=train_graph) as sess:
            self.model_loc = self._savedir + '/{}.ckpt'.format(self._name)
            sess.run(init)
            train_accuracy = []
            val_accuracy = []
            train_auc = []
            val_auc = []
            train_losses = []
            train_cats = []
            val_cats = []
            early_stopping = {'val_acc': -1.0, 'epoch': 0}

            print(110*'-')
            print('Train model: {}'.format(self.model_loc))
            print(110*'_')
            print('{:^25} | {:^25} | {:^25} | {:^25}'.format('Epoch', 'Training Loss', 
                'Training Accuracy', 'Validation Accuracy'))
            print(110*'-')

            cross_train_list = []
            cross_val_list = []
            weights_list = []
            
            train_start = time.time()
            for epoch in range(epochs):
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
                
                
                print('{:^25} | {:^25.4e} | {:^25.4f} | {:^25.4f}'.format(epoch + 1, 
                    train_losses[-1], train_accuracy[-1], val_accuracy[-1]))
                saver.save(sess, self.model_loc)
                cross_train_list.append(train_cross)
                cross_val_list.append(val_cross)
                train_cats.append(train_cat)
                val_cats.append(val_cat)


                if (self.enable_early=='yes'):
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
                        print(125*'-')
                        print('Early stopping invoked. '\
                                'Achieved best validation score of '\
                                '{:.4f} in epoch {}.'.format(
                                    early_stopping['val_acc'],
                                    early_stopping['epoch']+1))
                        best_epoch = early_stopping['epoch']
                        break
                else:
                    save_path = saver.save(sess, self.model_loc)


            print(110*'-')
            train_end=time.time()
            dtime = train_end - train_start

            self._write_parameters(epochs, batch_size, keep_prob, beta,
                    dtime, early_stopping, val_accuracy[-1])

            print('Model saved in: \n{}'.format(self._savedir))




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




    def _write_parameters(self, epochs, batch_size, keep_prob, beta, time,
            early_stop, val_acc_last):
        """Writes network parameters in a .txt. file
        """

        with open('{}/info.txt'.format(self._savedir),'w') as f:
            f.write('Date: {}\n'.format(datetime.datetime.now().strftime("%Y_%m_%d")))
            f.write('Time: {}\n'.format(datetime.datetime.now().strftime("%H_%M_%S")))
            f.write('Hidden layers: {}\n'.format(self._hidden_layers))
            f.write('Training Epochs: {}\n'.format(epochs))
            f.write('Batch Size: {}\n'.format(batch_size))
            f.write('Dropout: {}\n'.format(keep_prob))
            f.write('L2 Regularization: {}\n'.format(beta))
            f.write('Training Time: {} sec.\n'.format(time))
            f.write('Optimizer: {}\n'.format(self.optname))
            f.write('Initial learning rate: {}\n'.format(self.initial_learning_rate))
            f.write('Activation function: {}\n'.format(self.act_func))
            if (self.optimizer_options):
                f.write('Optimizer options: {}\n'.format(self.optimizer_options))
            f.write('Number of epochs trained: {}\n'.format(early_stop['epoch']))
            if (self.decay_learning_rate == 'yes'):
                f.write('Learning rate decay rate: {}\n'.format(self.decay_rate))
                f.write('Learning rate decay steps: {}\n'.format(self.decay_steps))
            if (self.batch_decay == 'yes'):
                f.write('Batch decay rate: {}\n'.format(self.batch_decay_rate))
                f.write('Batch decay steps: {}\n'.format(self.batch_decay_steps))
            if (self.enable_early == 'yes'):
                f.write('Early stopping interval: {}\n'.format(self.early_stop))
                f.write('Best validation epoch: {}\n'.format(early_stop['epoch']))
                f.write('Best validation accuracy: {}'.format(early_stop['val_acc']))
            else:
                f.write('Last validation accuracy: {}'.format(val_acc_last))
