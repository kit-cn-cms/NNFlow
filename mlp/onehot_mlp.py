# A one-hot output vector multi layer perceptron classifier. Currently depends on
# a custom dataset class defined in higgs_dataset.py. It is also assumed that
# there are no errors in the shape of the dataset.

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import datetime
import sys
import time
import pickle

from mlp.mlp import MLP

# from sklearn.metrics import roc_auc_score, roc_curve

class OneHotMLP(MLP):
    """A one-hot output vector classifier using a multi layer perceptron.

    Makes probability predictions on a set of features (a 1-dimensional numpy
    vector belonging either to the 'signal' or the 'background').
    """


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
#        self.sig_weight = sig_weight
#        self.bg_weight = bg_weight

        # check whether the model file exists
        if os.path.exists(self.savedir + '/{}.ckpt'.format(self._name)):
            self.trained = True
        else:
            self.trained = False

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



    def _model(self, data, W, B, keep_prob=1.0):
        """Model for the multi layer perceptron
        
        Arguments:
        ----------------
        data (tf.placeholder):
            A TensorFlow placeholder.
        W (list):
            A list with the TensorFlow Variables for the weights.
        B (list):
            A list with the TensorFlow Variables for the biases.

        Returns:
        ----------------
        out (tf.tensor)
            Prediction of the model.
        """

        self.act = self._get_activation_function()
        layer = tf.nn.dropout(self.act(tf.matmul(data, W[0]) + B[0]), 1)
        # if more the 1 hidden layer -> generate output via multiple weight
        # matrices 
        if len(self._hidden_layers) > 1:
            for weight, bias in zip(W[1:-1], B[1:-1]):
                layer = tf.nn.dropout(self.act(tf.matmul(layer, weight) +
                    bias), keep_prob)

        out = tf.matmul(layer, W[-1]) + B[-1]
        return out

    def train(self, train_data, val_data, optimizer='Adam', epochs = 10, batch_size = 100,
            learning_rate = 1e-3, keep_prob = 0.9, beta = 0.0, out_size=6, 
            optimizer_options=[], enable_early='no', early_stop=10, 
            decay_learning_rate='no', dlrate_options=[], batch_decay='no', 
            batch_decay_options=[], gpu_usage=None):
        """Trains the classifier

        Arguments:
        ----------------
        train_data (custom dataset):
            Contains training data.
        val_data (custom dataset):
            Contains validation data.
        optimizer (string):
            Name of the optimizer to be built.
        epochs (int): 
            Number of iterations over the whole training set.
        batch_size (int):
            Number of batches fed into one optimization step.
        learning_rate (float):
            Optimizer learning rate.
        keep_prob (float):
            Probability of a neuron to 'fire'.
        beta (float):
            L2 regularization coefficient; default 0.0 = regularization off.
        out_size (int):
            Dimension of output vector, i.e. number of classes.
        optimizer_options (list):
            List of additional options for the optimizer; can have different
            data types for different optimizers.
        enably_early (string):
            Check whether to use early stopping.
        early_stop (int):
            If validation accuracy does not increase over some epochs the training
            process will be ended and only the best model will be saved.
        decay_learning_rate (string):
            Indicates whether to decay the learning rate.
        dlrate_options (list):
            Options for exponential learning rate decay.
        batch_decay (string):
            Indicates whether to decay the batch size.
        batch_decay_options (list):
            Options for exponential batch size decay.
        """

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
            l2_reg = beta * self._l2_regularization(weights)
            # loss = tf.add(tf.reduce_mean(tf.reduce_sum(tf.mul(w, xentropy))), l2_reg, 
            #         name='loss')
            loss = tf.add(tf.reduce_mean(tf.multiply(w, xentropy)), l2_reg, 
                    name='loss')
            
            # optimizer
            optimizer, global_step = self._build_optimizer()
            train_step = optimizer.minimize(loss, global_step=global_step)

            # initialize all variables
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(weights + biases + [x_mean, x_std])
        
        
        sess_config = tf.ConfigProto()
        if gpu_usage['shared_machine']:
            if gpu_usage['restrict_visible_devices']:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_usage['CUDA_VISIBLE_DEVICES']

            if gpu_usage['allow_growth']:
                sess_config.gpu_options.allow_growth = True

            if gpu_usage['restrict_per_process_gpu_memory_fraction']:
                sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_usage['per_process_gpu_memory_fraction']

        with tf.Session(config=sess_config, graph=train_graph) as sess:
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
            times_list = []
            
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
                #print('train: {}'.format((train_corr, train_mistag)))
                train_accuracy.append(train_corr / (train_corr + train_mistag))
                
                val_pre = sess.run(yy_, {x:val_data.x})
                val_corr, val_mistag, val_cross, val_cat = self._validate_epoch(val_pre,
                        val_data.y, val_data.w)
                #print('validation: {}'.format((val_corr, val_mistag)))
                val_accuracy.append(val_corr / (val_corr + val_mistag))
                
                
                print('{:^25} | {:^25.4e} | {:^25.4f} | {:^25.4f}'.format(epoch + 1, 
                    train_losses[-1], train_accuracy[-1], val_accuracy[-1]))
                saver.save(sess, self.model_loc)
                cross_train_list.append(train_cross)
                cross_val_list.append(val_cross)
                train_cats.append(train_cat)
                val_cats.append(val_cat)

                if (epoch % 10 == 0):
                    t0 = time.time()
                    #self._plot_loss(train_losses)
                    #self._plot_accuracy(train_accuracy, val_accuracy, train_cats,
                    #        val_cats, epochs)
                    #self._plot_cross(train_cross, val_cross, epoch+1)
                    t1 = time.time()
                    times_list.append(t1 - t0)
                if (epoch == 0):
                    t0 = time.time()
                    app = '_{}'.format(epoch+1)
                    #self._write_list(train_pre, 'train_pred' + app)
                    #self._write_list(train_data.y, 'train_true' + app)
                    #self._write_list(val_pre, 'val_pred' + app)
                    #self._write_list(val_data.y, 'val_true' + app)
                    #self._plot_hists(train_pre, val_pre, train_data.y,
                    #        val_data.y, 1)
                    t1 = time.time()
                    times_list.append(t1 - t0)

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
                        t0 = time.time()
                        print(125*'-')
                        print('Early stopping invoked. '\
                                'Achieved best validation score of '\
                                '{:.4f} in epoch {}.'.format(
                                    early_stopping['val_acc'],
                                    early_stopping['epoch']+1))
                        best_epoch = early_stopping['epoch']
                        app = '_{}'.format(best_epoch+1)
                        self._plot_weight_matrices(weights_list[best_epoch],
                                best_epoch, early='yes')
                        self._plot_cross(cross_train_list[best_epoch],
                                cross_val_list[best_epoch], best_epoch,
                                early='yes')
                        self._plot_hists(best_train_pred, best_val_pred,
                                best_train_true, best_val_true, best_epoch+1)
                        self._write_list(best_train_pred, 'train_pred' + app)
                        self._write_list(best_train_true, 'train_true' + app)
                        self._write_list(best_val_pred, 'val_pred' + app)
                        self._write_list(best_val_true, 'val_true' + app)
                        t1 = time.time()
                        times_list.append(t1 - t0)
                        break
                else:
                    save_path = saver.save(sess, self.model_loc)


            print(110*'-')
            train_end=time.time()
            dtime = train_end - train_start - sum(times_list)

            self._plot_accuracy(train_accuracy, val_accuracy, train_cats,
                    val_cats, epochs)
            self._plot_loss(train_losses)
            self._write_parameters(epochs, batch_size, keep_prob, beta,
                    dtime, early_stopping, val_accuracy[-1])
            self._plot_weight_matrices(weights, epoch)
            self._plot_cross(train_cross, val_cross, epoch + 1)
            self._plot_hists(train_pre, val_pre, train_data.y, val_data.y,
                    epoch+1)
            self._plot_cross_dev(cross_train_list, cross_val_list, epoch+1)
            self._write_list(cross_train_list, 'train_cross')
            self._write_list(cross_val_list, 'val_cross')
            self._write_list(train_losses, 'train_losses')
            self._write_list(train_accuracy, 'train_accuracy')
            self._write_list(val_accuracy, 'val_accuracy')
            app = '_{}'.format(epoch+1)
            self._write_list(train_pre, 'train_pred' + app)
            self._write_list(train_data.y, 'train_true' + app)
            self._write_list(val_pre, 'val_pred' + app)
            self._write_list(val_data.y, 'val_true' + app)
            self.trained = True

            print('Model saved in: \n{}'.format(self._savedir))


    def _l2_regularization(self, weights):
        """Calculates and adds the squared values of the weights. This is used
        for L2 regularization.
        """
        # Applies tf.nn.l2_loss to all elements of weights
        # weights = map(lambda x: tf.nn.l2_loss(x), weights)
        # return sum(weights)
        losses = [tf.nn.l2_loss(w) for w in weights]
        return tf.add_n(losses)


    def _validate_epoch(self, pred, labels, weights):
        """Evaluates the training process.

        Arguments:
        ----------------
        pred (np.array):
            Predictions made by the model for the data fed into it.
        labels (np.array):
            Labels of the validation dataset.
        epoch (int):
            Training epoch.
        Returns:
        ----------------

        """

        arr_cross = np.zeros((self._number_of_output_neurons, self._number_of_output_neurons),dtype=np.float32)
        index_true = np.argmax(labels, axis=1)
        index_pred = np.argmax(pred, axis=1)
        for i in range(index_true.shape[0]):
            arr_cross[index_true[i]][index_pred[i]] += weights[i]
        correct = np.diagonal(arr_cross).sum()
        mistag = arr_cross.sum() - correct
        cat_acc = np.zeros((self._number_of_output_neurons), dtype=np.float32)
        for i in range(self._number_of_output_neurons): 
            cat_acc[i] = arr_cross[i][i] / (np.sum(arr_cross, axis=1)[i])

        
        return correct, mistag, arr_cross, cat_acc


    def _build_optimizer(self):
        self.initial_learning_rate = self.learning_rate
        """Returns a TensorFlow Optimizer.
        """
        global_step = tf.Variable(0, trainable=False)
        
        if (self.decay_learning_rate == 'yes'):
            try:
                self.decay_rate = self.decay_learning_rate_options[0]
            except IndexError:
                self.decay_rate = 0.97
            try:
                self.decay_steps = self.decay_learning_rate_options[1]
            except IndexError:
                self.decay_steps = 300
            self.learning_rate = (tf.train.exponential_decay(self.learning_rate,
                global_step, decay_rate=self.decay_rate, decay_steps=self.decay_steps))
        
        if (self.optname == 'Adam'):
            try:
                beta1 = self.optimizer_options[0]
            except IndexError:
                beta1 = 0.9
            try:
                beta2 = self.optimizer_options[1]
            except IndexError:
                beta2 = 0.999
            try:
                epsilon = self.optimizer_options[2]
            except IndexError:
                epsilon = 1e-8
            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1,
                    beta2=beta2, epsilon=epsilon)
            print('Building Adam Optimizer.')
            print('     learning_rate: {}'.format(self.learning_rate))
            print('     beta1: {}'.format(beta1))
            print('     beta2: {}'.format(beta2))
            print('     epsilon: {}'.format(epsilon))
        elif (self.optname == 'GradDescent'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            print('Building Gradient Descent Optimizer.')
            print('     learning_rate: {}'.format(self.learning_rate))
        elif (self.optname == 'Adagrad'):
            try:
                initial_accumulator_value = self.optimizer_options[0]
            except IndexError:
                initial_accumulator_value = 0.1
            optimizer = tf.train.AdagradOptimizer(self.learning_rate,
                    initial_accumulator_value=initial_accumulator_value)
            print('Building Adagrad Optimizer.')
            print('     learning_rate: {}'.format(self.learning_rate))
            print('     initial_accumulator_value: {}'
                    .format(initial_accumulator_value))
        elif (self.optname == 'Adadelta'):
            try:
                rho = self.optimizer_options[0]
            except IndexError:
                rho = 0.95
            try:
                epsilon = self.optimizer_options[1]
            except IndexError:
                epsilon = 1e-8
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate, rho=rho,
                epsilon=epsilon)
            print('Building Adadelta Optimizer.')
            print('     learning_rate: {}'.format(self.learning_rate))
            print('     rho: {}'.format(rho))
            print('     epsilon: {}'.format(epsilon))
        elif (self.optname == 'Momentum'):
            try:
                momentum = self.optimizer_options[0]
            except IndexError:
                momentum = 0.9
            try:
                use_nesterov = self.optimizer_options[1]
            except IndexError:
                use_nesterov = False
            optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                    momentum=momentum, use_nesterov=use_nesterov)
            print('Building Momentum Optimizer.')
            print('     initial learning_rate: {}'.format(self.initial_learning_rate))
            print('     momentum: {}'.format(momentum))
            print('     use_nesterov: {}'.format(use_nesterov))
        else:
            print('No Optimizer with name {} has been implemented.'
                    .format(self.optname))
            sys.exit('Aborting.')
        return optimizer, global_step




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
