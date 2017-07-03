from __future__ import absolute_import, division, print_function

import tensorflow as tf


class Optimizer(object):


    def __init__(self,
                 learning_rate
                 ):


        self._learning_rate       = learning_rate
        self._learning_rate_decay = False




    def get_optimizer_global_step(self):


        global_step = tf.Variable(0, trainable=False)


        if self._learning_rate_decay:
            decaying_learning_rate = tf.train.exponential_decay(self._learning_rate, global_step, decay_rate=self._learning_rate_decay_rate, decay_steps=self._learning_rate_dacay_steps)

            return self._optimizer_function(learning_rate = decaying_learning_rate, **self._optimizer_options_dict), global_step


        else:
            return self._optimizer_function(learning_rate = self._learning_rate,    **self._optimizer_options_dict), global_step




    def set_learning_rate_decay(self,
                                decay_rate,
                                decay_steps
                                ):


        self._learning_rate_decay       = True
        self._learning_rate_decay_rate  = decay_rate
        self._learning_rate_dacay_steps = decay_steps




class AdadeltaOptimizer(Optimizer):


    def __init__(self,
                 learning_rate,
                 rho           = 0.95,
                 epsilon       = 1e-08
                 ):


        Optimizer.__init__(self, learning_rate)


        self._optimizer_name     = 'Adadelta'
        self._optimizer_function = tf.train.AdadeltaOptimizer


        self._optimizer_options_dict = dict()

        self._optimizer_options_dict['rho']     = rho
        self._optimizer_options_dict['epsilon'] = epsilon




class AdagradOptimizer(Optimizer):


    def __init__(self,
                 learning_rate,
                 initial_accumulator_value = 0.1
                 ):


        Optimizer.__init__(self, learning_rate)


        self._optimizer_name     = 'Adagrad'
        self._optimizer_function = tf.train.AdagradOptimizer


        self._optimizer_options_dict = dict()

        self._optimizer_options_dict['initial_accumulator_value'] = initial_accumulator_value




class AdamOptimizer(Optimizer):


    def __init__(self,
                 learning_rate,
                 beta1         = 0.9,
                 beta2         = 0.999,
                 epsilon       = 1e-08
                 ):


        Optimizer.__init__(self, learning_rate)


        self._optimizer_name     = 'Adam'
        self._optimizer_function = tf.train.AdamOptimizer


        self._optimizer_options_dict = dict()

        self._optimizer_options_dict['beta1']   = beta1
        self._optimizer_options_dict['beta2']   = beta2
        self._optimizer_options_dict['epsilon'] = epsilon




class GradientDescentOptimizer(Optimizer):


    def __init__(self,
                 learning_rate
                 ):


        Optimizer.__init__(self, learning_rate)


        self._optimizer_name     = 'GradientDescent'
        self._optimizer_function = tf.train.GradientDescentOptimizer


        self._optimizer_options_dict = dict()




class MomentumOptimizer(Optimizer):


    def __init__(self,
                 learning_rate,
                 momentum,
                 use_nesterov  = False
                 ):


        Optimizer.__init__(self, learning_rate)


        self._optimizer_name     = 'Momentum'
        self._optimizer_function = tf.train.MomentumOptimizer


        self._optimizer_options_dict = dict()

        self._optimizer_options_dict['momentum']     = momentum
        self._optimizer_options_dict['use_nesterov'] = use_nesterov
