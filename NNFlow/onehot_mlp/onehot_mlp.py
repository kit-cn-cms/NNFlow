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

# from sklearn.metrics import roc_auc_score, roc_curve

class OneHotMLP:
    """A one-hot output vector classifier using a multi layer perceptron.

    Makes probability predictions on a set of features (a 1-dimensional numpy
    vector belonging either to the 'signal' or the 'background').
    """


    def __init__(self, n_features, h_layers, out_size, savedir, labels_text,
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

        self.n_features = n_features
        self.h_layers = h_layers
        self.out_size = out_size
        self.name = savedir.rsplit('/')[-1]
        self.savedir = savedir
        self.labels_text = labels_text
        self.branchlist = branchlist
#        self.sig_weight = sig_weight
#        self.bg_weight = bg_weight
        self.act_func = act_func

        # check whether the model file exists
        if os.path.exists(self.savedir + '/{}.ckpt'.format(self.name)):
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


    def _get_parameters(self):
        """Creates the TensorFlow Variables in two lists. 

        Returns:
        ----------------
        weights (list):
            A dictionary with the TensorFlow Variables for the weights.
        biases (list):
            A dictionary with the TensorFlow Variables for the biases.
        """

        n_features = self.n_features
        h_layers = self.h_layers

        weights = [tf.Variable(tf.random_normal([n_features, h_layers[0]], 
            stddev=tf.sqrt(2.0/n_features)), name = 'W_1')]
        biases = [tf.Variable(tf.zeros([h_layers[0]]), name = 'B_1')]


        # if more than 1 hidden layer -> create additional weights and biases
        if len(h_layers) > 1:
            for i in range(1, len(h_layers)):
                weights.append(tf.Variable(tf.random_normal([h_layers[i-1],
                    h_layers[i]], stddev = tf.sqrt(2.0 / h_layers[i-1])), name =
                    'W_{}'.format(i+1)))
                biases.append(tf.Variable(tf.zeros([h_layers[i]]), name =
                    'B_{}'.format(i+1)))

        # connect the last hidden layer to the output layer
        weights.append(tf.Variable(tf.random_normal([h_layers[-1], self.out_size],
            stddev = tf.sqrt(2.0/h_layers[-1])), name = 'W_out'))
        biases.append(tf.Variable(tf.zeros([self.out_size]), name = 'B_out'))
        
        return weights, biases

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

        self.act = self._build_activation_function()
        layer = tf.nn.dropout(self.act(tf.matmul(data, W[0]) + B[0]), 1)
        # if more the 1 hidden layer -> generate output via multiple weight
        # matrices 
        if len(self.h_layers) > 1:
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
        self._gpu_usage = gpu_usage

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
            x = tf.placeholder(tf.float32, [None, self.n_features], name='input')
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
        if 'shared_machine' in self._gpu_usage and self._gpu_usage['shared_machine']:
            if 'restrict_visible_devices' in self._gpu_usage and self._gpu_usage['restrict_visible_devices'] and 'CUDA_VISIBLE_DEVICES' in self._gpu_usage:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_usage['CUDA_VISIBLE_DEVICES']

            if 'allow_growth' in self._gpu_usage and self._gpu_usage['allow_growth']:
                sess_config.gpu_options.allow_growth = True

            if 'restrict_per_process_gpu_memory_fraction' in self._gpu_usage and self._gpu_usage['restrict_per_process_gpu_memory_fraction']:
                sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_usage['per_process_gpu_memory_fraction']

        with tf.Session(config=sess_config, graph=train_graph) as sess:
            self.model_loc = self.savedir + '/{}.ckpt'.format(self.name)
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
                        train_pre, train_data.y, train_data.w, epoch)
                #print('train: {}'.format((train_corr, train_mistag)))
                train_accuracy.append(train_corr / (train_corr + train_mistag))
                
                val_pre = sess.run(yy_, {x:val_data.x})
                val_corr, val_mistag, val_cross, val_cat = self._validate_epoch(val_pre,
                        val_data.y, val_data.w, epoch)
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
                    self._plot_loss(train_losses)
                    self._plot_accuracy(train_accuracy, val_accuracy, train_cats,
                            val_cats, epochs)
                    self._plot_cross(train_cross, val_cross, epoch + 1)
                    t1 = time.time()
                    times_list.append(t1 - t0)
                if (epoch == 0):
                    t0 = time.time()
                    app = '_{}'.format(epoch+1)
                    self._write_list(train_pre, 'train_pred' + app)
                    self._write_list(train_data.y, 'train_true' + app)
                    self._write_list(val_pre, 'val_pred' + app)
                    self._write_list(val_data.y, 'val_true' + app)
                    self._plot_hists(train_pre, val_pre, train_data.y,
                            val_data.y, 1)
                    t1 = time.time()
                    times_list.append(t1 - t0)

                if (self.enable_early=='yes'):
                    # Check for early stopping.
                    if (val_accuracy[-1] > early_stopping['val_acc']):
#                        save_path = saver.save(sess, self.model_loc)
                        save_path = saver.save(sess, self.model_loc.replace(".ckpt","_"+str(epoch)+".ckpt"))
                        print("saved model at epoch "+str(epoch))
                        best_train_pred = train_pre
                        best_train_true = train_data.y
                        best_val_pred = val_pre
                        best_val_true = val_data.y
                        early_stopping['val_acc'] = val_accuracy[-1]
                        early_stopping['epoch'] = epoch
                    elif ((epoch+1 - early_stopping['epoch']) > self.early_stop):
                        t0 = time.time()
                        print(125*'-')
                        print('Early stopping invoked. '\
                                'Achieved best validation score of '\
                                '{:.4f} in epoch {}.'.format(
                                    early_stopping['val_acc'],
                                    early_stopping['epoch']+1))
                        best_epoch = early_stopping['epoch']
                        app = '_{}'.format(best_epoch+1)
                        #self._plot_weight_matrices(weights_list[best_epoch],
                                #best_epoch, early='yes')
                        self._plot_cross(cross_train_list[best_epoch],
                                cross_val_list[best_epoch], best_epoch,
                                early='yes')
                        self._plot_hists(best_train_pred, best_val_pred,
                                best_train_true, best_val_true, best_epoch+1)
                        self._find_most_important_weights(weights_list[best_epoch],
                                n=30)
                        #self._write_list(best_train_pred, 'train_pred' + app)
                        #self._write_list(best_train_true, 'train_true' + app)
                        #self._write_list(best_val_pred, 'val_pred' + app)
                        #self._write_list(best_val_true, 'val_true' + app)
                        t1 = time.time()
                        times_list.append(t1 - t0)
                        break
                else:
                    save_path = saver.save(sess, self.model_loc)
                    print("saved model at epoch "+str(epoch))

            print(110*'-')
            train_end=time.time()
            dtime = train_end - train_start - sum(times_list)

            self._plot_accuracy(train_accuracy, val_accuracy, train_cats,
                    val_cats, epochs)
            self._plot_loss(train_losses)
            self._write_parameters(epochs, batch_size, keep_prob, beta,
                    dtime, early_stopping, val_accuracy[-1])
            #self._plot_weight_matrices(weights, epoch)
            self._plot_cross(train_cross, val_cross, epoch + 1)
            self._plot_hists(train_pre, val_pre, train_data.y, val_data.y,
                    epoch+1)
            self._plot_cross_dev(cross_train_list, cross_val_list, epoch+1)
            #self._write_list(cross_train_list, 'train_cross')
            #self._write_list(cross_val_list, 'val_cross')
            #self._write_list(train_losses, 'train_losses')
            #self._write_list(train_accuracy, 'train_accuracy')
            #self._write_list(val_accuracy, 'val_accuracy')
            app = '_{}'.format(epoch+1)
            #self._write_list(train_pre, 'train_pred' + app)
            #self._write_list(train_data.y, 'train_true' + app)
            #self._write_list(val_pre, 'val_pred' + app)
            #self._write_list(val_data.y, 'val_true' + app)
            if not (self.enable_early == 'yes'):
                self._find_most_important_weights(weights)
            self.trained = True

            print('Model saved in: \n{}'.format(self.savedir))


    def _l2_regularization(self, weights):
        """Calculates and adds the squared values of the weights. This is used
        for L2 regularization.
        """
        # Applies tf.nn.l2_loss to all elements of weights
        # weights = map(lambda x: tf.nn.l2_loss(x), weights)
        # return sum(weights)
        losses = [tf.nn.l2_loss(w) for w in weights]
        return tf.add_n(losses)


    def _validate_epoch(self, pred, labels, weights, epoch):
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

        arr_cross = np.zeros((self.out_size, self.out_size),dtype=np.float32)
        index_true = np.argmax(labels, axis=1)
        index_pred = np.argmax(pred, axis=1)
        for i in range(index_true.shape[0]):
            arr_cross[index_true[i]][index_pred[i]] += weights[i]
        correct = np.diagonal(arr_cross).sum()
        mistag = arr_cross.sum() - correct
        cat_acc = np.zeros((self.out_size), dtype=np.float32)
        for i in range(self.out_size): 
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


    def _build_activation_function(self):
        """Returns the activation function."""
        if (self.act_func == 'relu'):
            return tf.nn.relu
        elif (self.act_func == 'elu'):
            return tf.nn.elu
        elif (self.act_func == 'sigmoid'):
            return tf.sigmoid
        elif (self.act_func == 'softplus'):
            return tf.nn.softplus
        else:
            return tf.tanh


    def _write_parameters(self, epochs, batch_size, keep_prob, beta, time,
            early_stop, val_acc_last):
        """Writes network parameters in a .txt. file
        """

        with open('{}/info.txt'.format(self.savedir),'w') as f:
            f.write('Date: {}\n'.format(datetime.datetime.now().strftime("%Y_%m_%d")))
            f.write('Time: {}\n'.format(datetime.datetime.now().strftime("%H_%M_%S")))
            f.write('Hidden layers: {}\n'.format(self.h_layers))
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

    def _plot_loss(self, train_loss):
        """Plot loss of training and validation data.
        """
        plt.plot(train_loss, label='Training Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits=(-2,2))
        plt.legend(bbox_to_anchor=(1,1))
        plt.grid(True)
        plt.savefig(self.savedir + '/loss.pdf')
        plt.clf()


    def _plot_accuracy(self, train_accuracy, val_accuracy, train_cats, val_cats, epochs):
        """Plot the training and validation accuracies.
        """
        plt.plot(train_accuracy, color = 'red', label='Training accuracy')
        plt.plot(val_accuracy, color = 'black', label='Validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy development')
        plt.legend(loc='best')
        plt.grid(True)
        plt_name = self.name + '_accuracy'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.clf()
        
        arr = np.zeros((self.out_size, len(train_cats)))
        for i in range(len(train_cats)):
            for j in range(self.out_size):
                arr[j][i] = train_cats[i][j]
        for j in range(self.out_size):
            plt.plot(arr[j], label = self.labels_text[j])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Categories: Training Accuracy development')
        plt.legend(loc='best')
        plt.grid(True)
        plt_name = self.name + '_categories_train'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.clf()
        arr = np.zeros((self.out_size, len(val_cats)))
        for i in range(len(val_cats)):
            for j in range(self.out_size):
                arr[j][i] = val_cats[i][j]
        for j in range(self.out_size):
            plt.plot(arr[j], label = self.labels_text[j])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Categories: Validation Accuracy development')
        plt.legend(loc='best')
        plt.grid(True)
        plt_name = self.name + '_categories_val'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.clf()
    

    def _plot_cross(self, arr_train, arr_val, epoch, early='no'):
        arr_train_float = np.zeros((arr_train.shape[0], arr_train.shape[1]),
            dtype = np.float32)
        arr_val_float = np.zeros((arr_val.shape[0], arr_val.shape[1]), dtype =
                np.float32)
        arr_train_w_weights = np.zeros((arr_train.shape[0], arr_train.shape[1]),
            dtype = np.float32)
        arr_val_w_weights = np.zeros((arr_val.shape[0], arr_val.shape[1]), dtype
                = np.float32)
        for i in range(arr_train.shape[0]):
            row_sum = 0
            for j in range(arr_train.shape[1]):
                row_sum += arr_train[i][j]
            for j in range(arr_train.shape[1]):
                arr_train_float[i][j] = arr_train[i][j] / row_sum
#                if (i == 0):
#                    arr_train_w_weights[i][j] = 1.0 * arr_train[i][j] * self.sig_weight
#                else:
#                    arr_train_w_weights[i][j] = 1.0 * arr_train[i][j] * self.bg_weight
        for i in range(arr_val.shape[0]):
            row_sum = 0
            for j in range(arr_val.shape[1]):
                row_sum += arr_val[i][j]
            for j in range(arr_val.shape[1]):
                arr_val_float[i][j] = arr_val[i][j] / row_sum
#                if (i == 0):
#                    arr_val_w_weights[i][j] = 1.0 * arr_val[i][j] * self.sig_weight
#                else:
#                    arr_val_w_weights[i][j] = 1.0 * arr_val[i][j] * self.bg_weight
        if (early == 'yes'):
            epoch += 1
        #print(arr_train)
        #print('-----------------')
        #print(arr_val)
        x = np.linspace(0, self.out_size, self.out_size + 1)
        y = np.linspace(0, self.out_size, self.out_size + 1)
        xn, yn = np.meshgrid(x,y)
        cmap = matplotlib.cm.RdYlBu_r
        plt.pcolormesh(xn, yn, arr_train_float, cmap=cmap, vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self.labels_text)
        ax.set_yticklabels(self.labels_text)
        if (early=='yes'):
            plt.title('Heatmap: Training after early stopping in epoch {}'.format(epoch))
        else:
            plt.title("Heatmap: Training after epoch {}".format(epoch))
        plt.savefig(self.cross_savedir + '/{}_train.pdf'.format(epoch))
        plt.clf()
        cmap = matplotlib.cm.RdYlBu_r
        plt.pcolormesh(xn, yn, arr_val_float, cmap=cmap, vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self.labels_text)
        ax.set_yticklabels(self.labels_text)
        if (early=='yes'):
            plt.title('Heatmap: Validation after early stopping in epoch {}'.format(epoch))
        else:
            plt.title("Heatmap: Validation after epoch {}".format(epoch))
        plt.savefig(self.cross_savedir + '/{}_validation.pdf'.format(epoch))
        plt.clf()
        
        # Draw again with LogNorm colors
        cmap = matplotlib.cm.RdYlBu_r
        cmap.set_bad(color='white')
        minimum, maximum = find_limits(arr_train_float)
        plt.pcolormesh(xn, yn, arr_train_float, cmap=cmap,
                norm=colors.LogNorm(vmin=max(minimum, 1e-6), vmax=maximum))
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self.labels_text)
        ax.set_yticklabels(self.labels_text)
        if (early=='yes'):
            plt.title('Heatmap: Training after early stopping in epoch {}'.format(epoch))
        else:
            plt.title("Heatmap: Training after epoch {}".format(epoch))
        plt.savefig(self.cross_savedir + '/{}_train_colorlog.pdf'.format(epoch))
        plt.clf()
        cmap = matplotlib.cm.RdYlBu_r
        cmap.set_bad(color='white')
        minimum, maximum = find_limits(arr_val_float)
        plt.pcolormesh(xn, yn, arr_val_float, cmap=cmap,
                norm=colors.LogNorm(vmin=max(minimum,1e-6), vmax=maximum))
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self.labels_text)
        ax.set_yticklabels(self.labels_text)
        if (early=='yes'):
            plt.title('Heatmap: Validation after early stopping in epoch {}'.format(epoch))
        else:
            plt.title("Heatmap: Validation after epoch {}".format(epoch))
        plt.savefig(self.cross_savedir + '/{}_validation_colorlog.pdf'.format(epoch))
        plt.clf()

        # Draw again with absolute numbers
        cmap = matplotlib.cm.RdYlBu_r
        cmap.set_bad(color='white')
        minimum, maximum = find_limits(arr_train)
        plt.pcolormesh(xn, yn, arr_train, cmap=cmap, norm=colors.LogNorm(
            vmin=max(minimum, 1e-6), vmax=maximum))
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for yit in range(arr_train.shape[0]):
            for xit in range(arr_train.shape[1]):
                plt.text(xit + 0.5, yit + 0.5, '%d' % arr_train[yit, xit], 
                        horizontalalignment='center', verticalalignment='center',)
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self.labels_text)
        ax.set_yticklabels(self.labels_text)
        if (early=='yes'):
            plt.title('Heatmap: Training after early stopping in epoch {}'.format(epoch))
        else:
            plt.title("Heatmap: Training after epoch {}".format(epoch))
        plt.savefig(self.cross_savedir + '/{}_train_colorlog_absolute.pdf'.format(epoch))
        plt.clf()
        cmap = matplotlib.cm.RdYlBu_r
        cmap.set_bad(color='white')
        minimum, maximum = find_limits(arr_val)
        plt.pcolormesh(xn, yn, arr_val, cmap=cmap, norm=colors.LogNorm(
            vmin=max(minimum, 1e-6), vmax=maximum))
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for yit in range(arr_val.shape[0]):
            for xit in range(arr_val.shape[1]):
                plt.text(xit + 0.5, yit + 0.5, '%d' % arr_val[yit, xit], 
                        horizontalalignment='center', verticalalignment='center',)
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self.labels_text)
        ax.set_yticklabels(self.labels_text)
        if (early=='yes'):
            plt.title('Heatmap: Validation after early stopping in epoch {}'.format(epoch))
        else:
            plt.title("Heatmap: Validation after epoch {}".format(epoch))
        plt.savefig(self.cross_savedir + '/{}_validation_colorlog_absolute.pdf'.format(epoch))
        plt.clf()

        # Draw again with absolute numbers and weights
#        cmap = matplotlib.cm.RdYlBu_r
#        cmap.set_bad(color='white')
#        minimum, maximum = find_limits(arr_train_w_weights)
#        plt.pcolormesh(xn, yn, arr_train_w_weights, cmap=cmap, norm=colors.LogNorm(
#            vmin=max(minimum, 1e-6), vmax=maximum))
#        plt.colorbar()
#        plt.xlim(0, self.out_size)
#        plt.ylim(0, self.out_size)
#        
#        plt.xlabel("Predicted")
#        plt.ylabel("True")
#        for yit in range(arr_train_w_weights.shape[0]):
#            for xit in range(arr_train_w_weights.shape[1]):
#                plt.text(xit + 0.5, yit + 0.5, '%.2f' % arr_train_w_weights[yit, xit], 
#                        horizontalalignment='center', verticalalignment='center',)
#        ax = plt.gca()
#        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
#        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
#        ax.set_xticklabels(self.labels_text)
#        ax.set_yticklabels(self.labels_text)
#        if (early=='yes'):
#            plt.title('Heatmap: Training after early stopping in epoch {}'.format(epoch))
#        else:
#            plt.title("Heatmap: Training after epoch {}".format(epoch))
#        plt.savefig(self.cross_savedir + '/{}_train_colorlog_absolute_weights.pdf'.format(epoch))
#        plt.clf()
#        cmap = matplotlib.cm.RdYlBu_r
#        cmap.set_bad(color='white')
#        minimum, maximum = find_limits(arr_val_w_weights)
#        plt.pcolormesh(xn, yn, arr_val_w_weights, cmap=cmap, norm=colors.LogNorm(
#            vmin=max(minimum, 1e-6), vmax=maximum))
#        plt.colorbar()
#        plt.xlim(0, self.out_size)
#        plt.ylim(0, self.out_size)
#        
#        plt.xlabel("Predicted")
#        plt.ylabel("True")
#        for yit in range(arr_val_w_weights.shape[0]):
#            for xit in range(arr_val_w_weights.shape[1]):
#                plt.text(xit + 0.5, yit + 0.5, '%.2f' % arr_val_w_weights[yit, xit], 
#                        horizontalalignment='center', verticalalignment='center',)
#        ax = plt.gca()
#        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
#        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
#        ax.set_xticklabels(self.labels_text)
#        ax.set_yticklabels(self.labels_text)
#        if (early=='yes'):
#            plt.title('Heatmap: Validation after early stopping in epoch {}'.format(epoch))
#        else:
#            plt.title("Heatmap: Validation after epoch {}".format(epoch))
#        plt.savefig(self.cross_savedir + '/{}_validation_colorlog_absolute_weights.pdf'.format(epoch))
#        plt.clf()



    def _plot_weight_matrices(self, w, epoch, early='no'):
        np_weights = [weight.eval() for weight in w]
        self._write_list(np_weights[0], 'first_weights_{}'.format(epoch))
        self._write_list(np_weights, 'weights_{}'.format(epoch))
        for i in range(len(np_weights)):
            np_weight = np_weights[i]
            xr, yr = np_weight.shape
            x = range(xr + 1)
            y = range(yr + 1)
            xn, yn = np.meshgrid(y,x)
            plt.pcolormesh(xn, yn, np_weight)
            plt.colorbar()
            plt.xlim(0, np_weight.shape[1])
            plt.ylim(0, np_weight.shape[0])
            plt.xlabel('out')
            plt.ylabel('in')
            if (early=='yes'):
                title = 'Heatmap: weight[{}] after early stopping in epoch \
                {}'.format(i+1, epoch+1)
            else:
                title = "Heatmap: weight[{}], epoch: {}".format(i+1, epoch+1)
            plt.title(title)
            if (early=='yes'):
                plt.savefig(self.weights_savedir + 
                        '/epoch{}_weight{}_early.pdf'.format(epoch+1, i+1))
            else:
                plt.savefig(self.weights_savedir + 
                        '/epoch{}_weight{}.pdf'.format(epoch+1, i+1))
            plt.clf()



    def _write_list(self, outlist, name):
        """Writes a list of arrays into a name.txt file."""
        path = self.cross_savedir + '/' + name + '.txt'

        with open(path, 'wb') as out:
            pickle.dump(outlist, out)


    def _plot_hists(self, train_pred, val_pred, train_true, val_true, epoch):
        """Plot histograms of probability distributions.

        Arguments:
        ----------------
        train_pred (array):
            Array of shape (n_events_train, out_size) containing probabilities
            for each event to belong to each category.
        val_pred (array):
            Array of shape (n_events_val, out_size) containing probabilities for
            each event to belong to each category.
        train_true (array):
            Array of shape (n_events_train, out_size) containing the true
            training labels.
        val_true (array):
            Array of shape (n_events_val, out_size) containing the true
            validation labels.
        """
        hist_colors = ['navy', 'firebrick', 'darkgreen', 'purple', 'darkorange',
                'lightseagreen']
        print('Now drawing histograms') 
        bins = np.linspace(0,1,101)
        for i in range(train_pred.shape[1]):
            # sort the predicted values into the true categories. Just for
            # plotting. 
            for j in range(train_pred.shape[1]):
                arr = np.where(np.argmax(train_true, axis=1)==j)
                histo_list = np.transpose(train_pred[arr,i])
                # By the way, histo_list is an array
                if histo_list.size:
                    plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                            normed=True, histtype='step',label=self.labels_text[j])
            plt.xlabel('{} node output'.format(self.labels_text[i]))
            plt.ylabel('Arbitrary units.')
            plt.title('{} node output on training set'.format(self.labels_text[i]))
            plt.legend(loc='best')
            plt.savefig(self.hists_savedir_train + str(epoch) + '_' + str(i+1)+ '.pdf')
            plt.clf()
        for i in range(val_pred.shape[1]):
            for j in range(val_pred.shape[1]):
                arr = np.where(np.argmax(val_true, axis=1)==j)
                histo_list = np.transpose(val_pred[arr, i])
                if histo_list.size:
                    plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                            normed=True, histtype='step',label=self.labels_text[j])
            plt.xlabel('{} node output'.format(self.labels_text[i]))
            plt.ylabel('Arbitrary units.')
            plt.title('{} node output on validation set'.format(self.labels_text[i]))
            plt.legend(loc='best')
            plt.savefig(self.hists_savedir_val + str(epoch) + '_' + str(i+1)+ '.pdf')
            plt.clf()
        for i in range(train_pred.shape[1]):
            for j in range(train_pred.shape[1]):
                arr1 = (np.argmax(train_true, axis=1)==j)
                arr2 = (np.argmax(train_pred, axis=1)==i)
                arr = np.multiply(arr1, arr2)
                histo_list = train_pred[arr,i]
                if histo_list.size:
                    plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                            normed=True, histtype='step',label=self.labels_text[j])
            plt.xlabel('{} node output'.format(self.labels_text[i]))
            plt.ylabel('Arbitrary units.')
            plt.title('output for predicted {} on the training set'.format(self.labels_text[i]))
            plt.legend(loc='best')
            plt.savefig(self.hists_savedir_train + str(epoch) + '_' +
                    str(i+1)+'_predicted.pdf')
            plt.clf()
        for i in range(val_pred.shape[1]):
            for j in range(val_pred.shape[1]):
                arr1 = (np.argmax(val_true, axis=1)==j)
                arr2 = (np.argmax(val_pred, axis=1)==i)
                arr = np.multiply(arr1, arr2)
                histo_list = val_pred[arr,i]
                if histo_list.size:
                    plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                            normed=True, histtype='step',label=self.labels_text[j])
            plt.xlabel('{} node output'.format(self.labels_text[i]))
            plt.ylabel('Arbitrary units.')
            plt.title('output for predicted {} on the validation set'.format(self.labels_text[i]))
            plt.legend(loc='best')
            plt.savefig(self.hists_savedir_val + str(epoch) + '_' +
                    str(i+1)+'_predicted.pdf')
            plt.clf()
        # Draw again with logarithmic bins
        for i in range(train_pred.shape[1]):
            # sort the predicted values into the true categories. Just for
            # plotting. 
            for j in range(train_pred.shape[1]):
                arr = np.where(np.argmax(train_true, axis=1)==j)
                histo_list = np.transpose(train_pred[arr,i])
                if histo_list.size:
                    plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                            normed=True, histtype='step',label=self.labels_text[j],
                            log=True)
            plt.xlabel('{} node output'.format(self.labels_text[i]))
            plt.ylabel('Arbitrary units.')
            plt.title('{} node output on training set'.format(self.labels_text[i]))
            plt.legend(loc='best')
            plt.savefig(self.hists_savedir_train + str(epoch) + '_' + str(i+1)+
                    '_log.pdf')
            plt.clf()
        for i in range(val_pred.shape[1]):
            for j in range(val_pred.shape[1]):
                arr = np.where(np.argmax(val_true, axis=1)==j)
                histo_list = np.transpose(val_pred[arr,i])
                if histo_list.size:
                    plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                            normed=True, histtype='step',label=self.labels_text[j],
                            log=True)
            plt.xlabel('{} node output'.format(self.labels_text[i]))
            plt.ylabel('Arbitrary units.')
            plt.title('{} node output on validation set'.format(self.labels_text[i]))
            plt.legend(loc='best')
            plt.savefig(self.hists_savedir_val + str(epoch) + '_' + str(i+1)+
                    '_log.pdf')
            plt.clf()
        for i in range(train_pred.shape[1]):
            for j in range(train_pred.shape[1]):
                arr1 = (np.argmax(train_true, axis=1)==j)
                arr2 = (np.argmax(train_pred, axis=1)==i)
                arr = np.multiply(arr1, arr2)
                histo_list = train_pred[arr,i]
                if histo_list.size:
                    plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                            normed=True, histtype='step',label=self.labels_text[j],
                            log=True)
            plt.xlabel('{} node output'.format(self.labels_text[i]))
            plt.ylabel('Arbitrary units.')
            plt.title('output for predicted {} on the training set'.format(self.labels_text[i]))
            plt.legend(loc='best')
            plt.savefig(self.hists_savedir_train + str(epoch) + '_' +
                    str(i+1)+'_predicted_log.pdf')
            plt.clf()
        for i in range(val_pred.shape[1]):
            for j in range(val_pred.shape[1]):
                arr1 = (np.argmax(val_true, axis=1)==j)
                arr2 = (np.argmax(val_pred, axis=1)==i)
                arr = np.multiply(arr1, arr2)
                histo_list = val_pred[arr,i]
                if histo_list.size:
                    plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                            normed=True, histtype='step',label=self.labels_text[j],
                            log=True)
            plt.xlabel('{} node output'.format(self.labels_text[i]))
            plt.ylabel('Arbitrary units.')
            plt.title('output for predicted {} on the validation set'.format(self.labels_text[i]))
            plt.legend(loc='best')
            plt.savefig(self.hists_savedir_val + str(epoch) + '_' +
                    str(i+1)+'_predicted_log.pdf')
            plt.clf()
        print('Done')


    def _plot_cross_dev(self, train_list, val_list, epoch):
        """Plot the development of misclassifications.

        Arguments:
        ----------------
        train_list (list):
            List containing a cross check array for each epoch.
        val_list (list):
            List containing a cross check array for each epoch.
        epoch (int):
            Epoch.
        """
        train_x_classified_as_y = np.zeros((epoch, self.out_size, self.out_size))
        val_x_classified_as_y = np.zeros((epoch, self.out_size, self.out_size))
        train_y_classified_as_x = np.zeros((epoch, self.out_size,
            self.out_size))
        val_y_classified_as_x = np.zeros((epoch, self.out_size, self.out_size))


        for list_index in range(len(train_list)):
            arr_train = train_list[list_index]
            arr_val = val_list[list_index]
            row_sum_train = np.sum(arr_train, axis=1)
            column_sum_train = np.sum(arr_train, axis=0)
            row_sum_val = np.sum(arr_val, axis=1)
            column_sum_val = np.sum(arr_val, axis=0)
            for i in range(arr_train.shape[0]):
                for j in range(arr_train.shape[1]):
                    if (row_sum_train[i] != 0):
                        train_x_classified_as_y[list_index,i,j] = arr_train[i][j] / row_sum_train[i]
                    else:
                        train_x_classified_as_y[list_index,i,j] = arr_train[i][j]
                    if (row_sum_val[i] != 0):
                        val_x_classified_as_y[list_index,i,j] = arr_val[i][j] / row_sum_val[i]
                    else:
                        val_x_classified_as_y[list_index,i,j] = arr_val[i][j]
                    if (column_sum_train[j] != 0):
                        train_y_classified_as_x[list_index,i,j] = arr_train[i][j] / column_sum_train[j]
                    else:
                        train_y_classified_as_x[list_index,i,j] = arr_train[i][j]
                    if (column_sum_val[j] != 0):
                        val_y_classified_as_x[list_index,i,j] = arr_val[i][j] / column_sum_val[j]
                    else:
                        val_y_classified_as_x[list_index,i,j] = arr_val[i][j]

        for i in range(train_x_classified_as_y.shape[1]):
            for j in range(train_x_classified_as_y.shape[2]):
                plt.plot(train_x_classified_as_y[:,i,j],
                        label=self.labels_text[j])
            plt.title(self.labels_text[i] + ' classified as')
            plt.xlabel('Epoch')
            plt.ylabel('Tagging rate')
            plt.xlim(0, epoch)
            plt.ylim(0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')
            plt.savefig(self.mistag_savedir + 'train_x_{}_as.pdf'.format(i))
            # plt.savefig(self.mistag_savedir + 'train_x_{}_as.png'.format(i))
            # plt.savefig(self.mistag_savedir + 'train_x_{}_as.eps'.format(i))
            plt.clf()
        for i in range(val_x_classified_as_y.shape[1]):
            for j in range(val_x_classified_as_y.shape[2]):
                plt.plot(val_x_classified_as_y[:,i,j],
                        label=self.labels_text[j])
            plt.title(self.labels_text[i] + ' classified as')
            plt.xlabel('Epoch')
            plt.ylabel('Tagging rate')
            plt.xlim(0, epoch)
            plt.ylim(0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')
            plt.savefig(self.mistag_savedir + 'val_x_{}_as.pdf'.format(i))
            # plt.savefig(self.mistag_savedir + 'val_x_{}_as.png'.format(i))
            # plt.savefig(self.mistag_savedir + 'val_x_{}_as.eps'.format(i))
            plt.clf()
        for i in range(train_y_classified_as_x.shape[2]):
            for j in range(train_y_classified_as_x.shape[1]):
                plt.plot(train_y_classified_as_x[:,i,j],
                        label=self.labels_text[j])
            plt.title('classified as ' + self.labels_text[i])
            plt.xlabel('Epoch')
            plt.ylabel('Tagging rate')
            plt.xlim(0, epoch)
            plt.ylim(0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')
            plt.savefig(self.mistag_savedir + 'train_as_x_{}.pdf'.format(i))
            # plt.savefig(self.mistag_savedir + 'train_as_x_{}.png'.format(i))
            # plt.savefig(self.mistag_savedir + 'train_as_x_{}.eps'.format(i))
            plt.clf()
        for i in range(val_y_classified_as_x.shape[2]):
            for j in range(val_y_classified_as_x.shape[1]):
                plt.plot(train_y_classified_as_x[:,i,j],
                        label=self.labels_text[j])
            plt.title('classified as ' + self.labels_text[i])
            plt.xlabel('Epoch')
            plt.ylabel('Tagging rate')
            plt.xlim(0, epoch)
            plt.ylim(0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')
            plt.savefig(self.mistag_savedir + 'val_as_x_{}.pdf'.format(i))
            # plt.savefig(self.mistag_savedir + 'val_as_x_{}.png'.format(i))
            # plt.savefig(self.mistag_savedir + 'val_as_x_{}.eps'.format(i))
            plt.clf()

    def _find_most_important_weights(self, w, n=10):
        # Only consider the first hidden layer.
        weight = w[0].eval()
        weight = np.absolute(weight)
        weight_abs_mean = np.mean(weight, axis=1, dtype=np.float32)
        with open(self.branchlist, 'r') as f:
            self.branches = [line.strip() for line in f]
        max_weight = 0.0
        values = []
        indices = []
        branchnames = []

        for i in range(n):
            index = np.argmax(weight_abs_mean)
            indices.append(index)
            values.append(weight_abs_mean[index])
            branchnames.append(self.branches[index])
            weight_abs_mean[index] = 0.0


        with open (self.cross_savedir + '/most_important_variables.txt', 'w') as f:
            for i in range(n):
                f.write('branch: {}, mean_abs: {}\n'.format(branchnames[i],
                    values[i]))


def find_limits(arr):
    minimum = np.min(arr) / (np.pi**2.0 * np.exp(1.0)**2.0)
    maximum = np.max(arr) * np.pi**2.0 * np.exp(1.0)**2.0
    return minimum, maximum
