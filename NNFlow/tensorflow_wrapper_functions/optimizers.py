from __future__ import absolute_import, division, print_function

import tensorflow as tf


class Optimizer(object):


    def __init__(self,
                 optimizer_name,
                 learning_rate
                 ):


        self._optimizer_name = optimizer_name
        self._learning_rate  = learning_rate

        self._learning_rate_decay = False




    def set_learning_rate_decay(self,
                                decay_rate,
                                decay_steps
                                ):


        self._learning_rate_decay       = True
        self._learning_rate_decay_rate  = decay_rate
        self._learning_rate_dacay_steps = decay_steps




    def get_optimizer_global_step(self):


        global_step = tf.Variable(0, trainable=False)


        if self._learning_rate_decay:
            learning_rate = tf.train.exponential_decay(self._learning_rate, global_step, decay_rate=self._learning_rate_decay_rate, decay_steps=self._learning_rate_dacay_steps)

        else:
            learning_rate = self._learning_rate
