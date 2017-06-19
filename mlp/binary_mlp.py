# A Binary Multilayerperceptron Classifier. Currently Depends on a custom
# dataset class defined in data_frame.py.
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import time

from sklearn.metrics import roc_auc_score, roc_curve

from mlp.mlp import MLP

class BinaryMLP(MLP):
    """A Binary Classifier using a Multilayerperceptron.

    Makes probability predictions on a set of features (A 1-dimensional
    numpy vector belonging either to the 'signal' or the 'background'.

    Arguments:
    ----------------
    n_variables (int) :
    The number of input features.
    h_layers (list):
    A list representing the hidden layers. Each entry gives the number of
    neurons in the equivalent layer.
    savedir (str) :
    Path to the directory the model should be saved in.
    activation (str) :
    Default is 'relu'. Activation function used in the model. Also possible 
    is 'tanh' or 'sigmoid'.
    var_names (str) :
    Optional. If given this string is plotted in the controll plots title.
    Should describe the used dataset. 
    """

    def init_old(self, n_variables, h_layers, savedir, activation,
                 var_names=None):
        self.variables = var_names

        # check wether model file exists
        if os.path.exists(self.savedir + '/{}.ckpt.meta'.format(self._name)):
            self.trained = True
        else:
            self.trained = False

        # create save directory if needed
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

   
            
    def _get_optimizer(self):
        """Get Opitimizer to be used for minimizing the loss function.

        Returns:
        ------------
        opt (tf.train.Optimizer) :
        Chosen optimizer.
        global_step (tf.Variable) :
        Variable that counts the minimization steps. Not trainable, only used 
        for internal bookkeeping.
        """
        
        global_step = tf.Variable(0, trainable=False)
        if self._lr_decay:
            learning_rate = (tf.train
                             .exponential_decay(self._lr, global_step,
                                                decay_steps = self._lr_decay[1],
                                                 decay_rate = self._lr_decay[0])
                             )
        else:
            learning_rate = self._lr

        if self._optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif self._optimizer =='gradientdescent':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif self._optimizer == 'momentum':
            if self._momentum:
                opt = tf.train.MomentumOptimizer(learning_rate,
                                                 self._momentum[0],
                                                 use_nesterov = self._momentum[1])
            else:
                sys.exit('No momentum term for "momentum" optimizer available.')
        else :
            sys.exit('Choose Optimizer: "adam", ' +
            '"gradientdescent" or "momentum"')

        return opt, global_step
    
    def _model(self, x, W, B, keep_prob=1.0):
        """Model for the multi layer perceptron

        Arguments:
        --------------
        data (tf.placeholder) : 
        A tensorflow placeholder.
        W (list) :
        A list with the tensorflow Variables for the weights.
        B (list) :
        A list with the tensorflow Variables for the biases.

        Returns:
        ---------------
        output_node (tf.tensor)
        Prediction of the model.
        """
        activation = self._get_activation_function()
        
        layer = tf.nn.dropout(activation(
            tf.add(tf.matmul(x, W[0]), B[0])), 1.)
        
        # fancy loop for creating hidden layer
        if len(self._hidden_layers) > 1:
            for weight, bias in zip(W[1:-1], B[1:-1]):
                layer = tf.nn.dropout(activation(
                    tf.add(tf.matmul(layer, weight), bias)),keep_prob)

        logit = tf.add(tf.matmul(layer, W[-1]), B[-1]) 

        return logit

    def train(self, train_data, val_data, epochs, batch_size,
              lr, optimizer, early_stop, keep_prob, beta, gpu_usage,
              momentum=None, lr_decay=None):
        """Train Neural Network with given training data set.

        Arguments:
        -------------
        train_data (custom dataset) :
        Contains training data.
        val_data (custom dataset) :
        Contains validation data.
        savedir (string) :
        Path to directory to save Plots.
        epochs (int) :
        Number of iterations over the whole trainig set.
        batch_size (int) :
        Number of batches fed into on optimization step.
        lr (float) :
        Default is 1e-3. Learning rate use by the optimizer for minizing the
        loss. The default value should work with 'adam'. Other optimizers may
        require other values. Tweak the learning rate for best results.
        optimizer (str) :
        Default is 'Adam'. Other options are 'gradientdescent' or 'momentum'.
        momentum list(float, bool) :
        Default is [0.1, True]. Only used if 'momentum' is used as optimizer.
        momentum[0] is the momentum, momentum[1] indicates, wether Nesterov
        momentum should be used.
        lr_decay list() :
        Default is None. Only use if 'gradientdescent' or 'momentum' is used as
        optimizer. List requires form [decay_rate (float), decay_steps (int)].
        List parameters are used to for exponentailly decaying learning rate.
        Try [0.96, 100000].
        early_stop (int):
        Default is 20. If Validation AUC has not increase  in the given number
        of epochs, the training is stopped. Only the model with the highest 
        validation auc score is saved.
        keep_prob (float):
        Probability of a neuron to 'activate'.
        beta (float):
        L2 regularization coefficient. Defaul 0.0 = regularization off.
        """
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
            l2_reg = beta*self._l2_regularization(weights)
            loss = tf.add(tf.reduce_mean(tf.multiply(w, xentropy)), l2_reg,
                                  name='loss')

            # optimizer
            opt, global_step = self._get_optimizer()
            train_step = opt.minimize(loss, global_step=global_step)

            # initialize the variables
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(weights + biases + [x_mean, x_std])

        

        config = tf.ConfigProto()
        if gpu_usage['shared_machine']:
            if gpu_usage['restrict_visible_devices']:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_usage['CUDA_VISIBLE_DEVICES']
            
            if gpu_usage['allow_growth']:
                config.gpu_options.allow_growth = True
            
            if gpu_usage['restrict_per_process_gpu_memory_fraction']:
                config.gpu_options.per_process_gpu_memory_fraction = gpu_usage['per_process_gpu_memory_fraction']
        
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

            # get training prediction for validation, use batches to prevent
            # allocating to much gpu memory
            train_data.shuffle()
            evts_per_batch = 100
            total_batches = int(train_data.n/evts_per_batch)
            train_pre = []
            for batch in range(0, total_batches):
                train_x, _, _ = train_data.next_batch(evts_per_batch)
                pred = sess.run(yy_, {x: train_x})
                train_pre.append(pred)
            train_pre = np.concatenate(train_pre, axis=0)
            print(90*'-')
            self._validation(train_pre, train_data.y[:total_batches*evts_per_batch],
                             early_stopping['val_pre'], val_data.y)
            self._plot_auc_dev(train_auc, val_auc, early_stopping['epoch'])
            self._plot_loss(train_loss)
            self.trained = True
            self._write_parameters(batch_size, keep_prob, beta,
                                   np.mean(epoch_durations), early_stopping)
            print('Model saved in: {}'.format(save_path))
            print(90*'-')
            
    def _l2_regularization(self, weights):
        """Calculate and adds the squared values of the weights. This is used
        for L2 Regularization.
        """
        weights = map(lambda x: tf.nn.l2_loss(x), weights)

        return tf.add_n(weights)




    def _write_parameters(self, batch_size, keep_prob, beta, time,
                          early_stop):
        """Writes network parameters in a .txt file
        """

        with open('{}/NN_Info.txt'.format(self._savedir), 'w') as f:
            f.write('Number of input variables: {}\n'.format(self._number_of_input_neurons))
            f.write('Number of hidden layers and neurons: {}\n'
                    .format(self._hidden_layers))
            f.write('Activation function: {}\n'.format(self.activation))
            f.write('Optimizer: {}, Learning Rate: {}\n'
                    .format(self._optimizer, self._lr))
            if self._momentum:
                f.write('Momentum: {}, Nesterov: {}\n'
                        .format(self._momentum[0], self._momentum[1]))
            if self._lr_decay:
                f.write('Decay rate: {}, Decay steps: {}\n'
                        .format(self._lr_decay[0], self._lr_decay[1]))
            f.write('Number of epochs trained: {}\n'
                    .format(early_stop['epoch']))
            f.write('Validation ROC-AUC score: {:.4f}\n'
                    .format(early_stop['auc']))
            f.write('Batch Size: {}\n'.format(batch_size))
            f.write('Dropout: {}\n'.format(keep_prob))
            f.write('L2 Regularization: {}\n'.format(beta))
            f.write('Mean Training Time per Epoch: {} s\n'.format(time))

        with open('{}/NN_Info.txt'.format(self._savedir), 'r') as f:
            for line in f:
                print(line)
        print(90*'-')

    def _validation(self, t_pred, t_labels, v_pred, v_labels):
        """Validation of the training process.
        Makes plots of ROC curves and displays the development of AUC score.

        Arguments:
        ----------------
        pred (np.array, shape(-1,)) :
        Predictions for data put in the model.
        labels (np.array, shape(-1)) :
        Lables of the validation dataset.
        """
        
        def seperate_sig_bkg(pred, labels):
            """This functions seperates signal and background output of the
            neural network.
            """
            y = np.hstack((pred, labels))
            sig = y[y[:,1]==1, 0]
            bg = y[y[:,1]==0, 0]
            return sig, bg
        
        # plot distribution
        t_sig, t_bg = seperate_sig_bkg(t_pred, t_labels)
        v_sig, v_bg = seperate_sig_bkg(v_pred, v_labels)
        bin_edges = np.linspace(0, 1, 30)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        v_sig = np.histogram(v_sig, bins=bin_edges, normed=True)[0]
        v_bg = np.histogram(v_bg, bins=bin_edges, normed=True)[0]
        
        plt.hist(t_sig, bins=bin_edges, histtype='step', lw=1.5,
                 label='Signal (Training)', normed='True', color='#1f77b4')
        plt.hist(t_bg, bins=bin_edges, lw=1.5, histtype='step',
                 label='Untergrund (Training)', normed='True',
                 color='#d62728')
        plt.plot(bin_centres, v_sig, ls='', marker='o', markersize=3, color='#1f77b4',
                 label='Signal (Validierung)')
        plt.plot(bin_centres, v_bg, ls='', marker='o', markersize=3, color='#d62728',
                 label='Untergrund (Validierung)')
        plt.ylim([0.0, np.amax(v_sig)*1.4])
        
        plt.legend(loc='upper left')
        plt.xlabel('Netzwerk Ausgabe')
        plt.ylabel('Ereignisse (normiert)')
        if self.variables:
            plt.title(self.variables, loc='left')
        plt.title(self._name, loc='center')
        # plt.title('CMS Private Work', loc='right')
        
        
        plt_name = self._name + '_dist'
        plt.savefig(self._savedir + '/' + plt_name + '.pdf')
        plt.savefig(self._savedir + '/' + plt_name + '.png')
        plt.savefig(self._savedir + '/' + plt_name + '.eps')
        plt.clf()

        # roc curve
        v_pred = np.reshape(v_pred, -1)
        fpr, tpr, thresh = roc_curve(v_labels, v_pred)
        auc = roc_auc_score(v_labels, v_pred)
        #plot the roc_curve
        plt_name = self._name +  '_roc'
        
        plt.plot(tpr, np.ones(len(fpr)) - fpr, color='#1f77b4',
                 label='ROC Kurve (Integral = {:.4f})'.format(auc), lw=1.7)
        #make the plot nicer
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        
        plt.xlabel('Signaleffizienz')
        plt.ylabel('Untergrundablehnung')
        if self.variables:
            plt.title(self.variables, loc='left')
        plt.grid(True)
        plt.title(self._name, loc='center')
        # plt.title('CMS Private Work', loc='right')
        plt.legend(loc='best')
        
        plt.savefig(self._savedir + '/' + plt_name + '.pdf')
        plt.savefig(self._savedir + '/' + plt_name + '.png')
        plt.savefig(self._savedir + '/' + plt_name + '.eps')
        plt.clf()
            
    def _plot_loss(self, train_loss):
        """Plot loss of training and validation data.
        """
        
        plt.plot(train_loss, label= 'Trainingsfehler', color='#1f77b4', ls='',
                 marker='^')
        plt.xlabel('Epoche')
        plt.ylabel('Fehlerfunktion')
        
        if self.variables:
            plt.title(self.variables, loc='left')
        plt.title(self._name, loc='center')
        # plt.title('CMS Private Work', loc='right')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        plt.legend(loc=0)
        
        plt_name = self._name + '_loss'
        plt.savefig(self._savedir + '/' + plt_name + '.pdf')
        plt.savefig(self._savedir + '/' + plt_name + '.png')
        plt.savefig(self._savedir + '/' + plt_name + '.eps')
        plt.clf()

    def _plot_auc_dev(self, train_auc, val_auc, stop):
        """Plot ROC-AUC-Score development
        """
        plt.plot(range(1, len(train_auc)+1), train_auc, color='#1f77b4', label='Training', ls='',
                 marker='^')
        plt.plot(range(1, len(val_auc)+1), val_auc, color='#ff7f0e', label='Validierung', ls='',
                 marker='^')
        
        # make plot nicer
        plt.xlabel('Epoche')
        plt.ylabel('ROC Integral')
        plt.axvline(x=stop, color='r')
        if self.variables:
            plt.title(self.variables, loc='left')
        plt.title(self._name, loc='center')
        # plt.title('CMS Private Work', loc='right')
        plt.legend(loc='best', frameon=False)

        # save plot
        plt_name = self._name + '_auc_dev'
        plt.savefig(self._savedir + '/' + plt_name + '.pdf')
        plt.savefig(self._savedir + '/' + plt_name + '.png')
        plt.savefig(self._savedir + '/' + plt_name + '.eps')
        plt.clf()
