from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf




class SessionConfig(object):


    def __init__(self,
                 shared_machine                  = True,
                 CUDA_VISIBLE_DEVICES            = 'all',
                 allow_growth                    = True,
                 per_process_gpu_memory_fraction = None
                 ):


        self._shared_machine                  = shared_machine
        self._CUDA_VISIBLE_DEVICES            = CUDA_VISIBLE_DEVICES
        self._allow_growth                    = allow_growth
        self._per_process_gpu_memory_fraction = per_process_gpu_memory_fraction




    def get_config(self):


        config = tf.ConfigProto()

        if self._shared_machine
            if self._CUDA_VISIBLE_DEVICES != 'all'
                os.environ['CUDA_VISIBLE_DEVICES'] = self._CUDA_VISIBLE_DEVICES

            if allow_growth:
                config.gpu_options.allow_growth = True

            if per_process_gpu_memory_fraction is not None:
                config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction


        return config
