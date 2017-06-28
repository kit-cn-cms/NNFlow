from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf


class NNFlowBaseClass(object):


    def _get_session_config(self,
                            gpu_usage
                            ):
             
             
        config = tf.ConfigProto()
        if gpu_usage['shared_machine']:
            if gpu_usage['restrict_visible_devices']:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_usage['CUDA_VISIBLE_DEVICES']
             
            if gpu_usage['allow_growth']:
                config.gpu_options.allow_growth = True
             
            if gpu_usage['restrict_per_process_gpu_memory_fraction']:
                config.gpu_options.per_process_gpu_memory_fraction = gpu_usage['per_process_gpu_memory_fraction']
             
             
        return config
