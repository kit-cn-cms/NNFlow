from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf




class SessionConfig(object):


    def __init__(self,
                 visible_devices                 = 'all',
                 allow_growth                    = True,
                 per_process_gpu_memory_fraction = None
                 ):


        self._visible_devices                 = visible_devices
        self._allow_growth                    = allow_growth
        self._per_process_gpu_memory_fraction = per_process_gpu_memory_fraction




    def get_tf_config(self):


        tf_config = tf.ConfigProto()

        if self._visible_devices != 'all':
            os.environ['CUDA_VISIBLE_DEVICES'] = self._visible_devices

        if self._allow_growth:
            tf_config.gpu_options.allow_growth = True

        if self._per_process_gpu_memory_fraction is not None:
            tf_config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction


        return tf_config
