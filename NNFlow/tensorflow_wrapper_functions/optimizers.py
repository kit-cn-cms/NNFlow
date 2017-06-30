from __future__ import absolute_import, division, print_function

import tensorflow as tf


class Optimizer(object):


    def __init__(self,
                 learning_rate
                 ):


        self._learning_rate  = learning_rate

        self._learning_rate_decay = False




    def set_learning_rate_decay(self,
                                decay_rate,
                                decay_steps
                                ):


        self._learning_rate_decay       = True
        self._learning_rate_decay_rate  = decay_rate
        self._learning_rate_dacay_steps = decay_steps




    def _get_learning_rate(self):


        global_step = tf.Variable(0, trainable=False)


        if self._learning_rate_decay:
            learning_rate = tf.train.exponential_decay(self._learning_rate, global_step, decay_rate=self._learning_rate_decay_rate, decay_steps=self._learning_rate_dacay_steps)

        else:
            learning_rate = self._learning_rate


        return learning_rate




class AdadeltaOptimizer(Optimizer):


    def __init__(self,
                 rho,
                 epsilon
                 ):


        Optimizer.__init__(self, learning_rate)

        self._rho     = rho
        self._epsilon = epsilon




    def get_optimizer_global_step(self):


        global_step = tf.Variable(0, trainable=False)
        learning_rate = self._get_learning_rate()

        return tf.train.AdadeltaOptimizer(learning_rate, rho=rho, epsilon=epsilon)




class AdamOptimizer(Optimizer):


    def __init__(self,
                 learning_rate,
                 beta1,
                 beta2,
                 epsilon
                 ):


        Optimizer.__init__(self, learning_rate)

        self._beta1   = beta1
        self._beta2   = beta2
        self._epsilon = epsilon




    def get_optimizer_global_step(self):


        global_step = tf.Variable(0, trainable=False)
        learning_rate = self._get_learning_rate()

        tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
