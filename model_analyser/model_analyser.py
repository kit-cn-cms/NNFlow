from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd

import tensorflow as tf


class ModelAnalyser(object):
    
    def __init__(self,
                 path_to_model,
                 gpu_usage):


        self._path_to_model = path_to_model
        self._path_to_data = path_to_data
        self._gpu_usage = gpu_usage




    def save_variable_ranking(self,
                              save_dir,
                              path_to_variablelist):


        gpu_config = self._get_gpu_config()
        graph = tf.Graph()
        with tf.Session(config=config, graph=graph) as sess:
             saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
             saver.restore(sess, path_to_model)
             weights = graph.get_tensor_by_name("W_1:0").eval()

        weight_abs = np.absolute(weights)
        weight_abs_mean = np.mean(weight_abs, axis=1)
        
        with open(path_to_variablelist, 'r') as file_variablelist:
            variablelist = [variable.rstrip() for variable in file_variablelist.readlines()]

        variable_ranking = pd.Series(weight_abs_mean, index=variablelist)
        variable_ranking.sort_values(inplace=True)

        with open(os.path.join(save_dir, 'variable_ranking.txt'), 'w') as outfile:
            for variable in variable_ranking.index:
                outfile.write(variable + (60-len(variable) if len(variable)<60 else 1)*' ' + str(variable_ranking[variable]) + '\n')

        variable_ranking.to_msgpack(os.path.join(save_dir, 'variable_ranking.msg'))



    def _get_gpu_config(self):
        
        config = tf.ConfigProto()
        if self._gpu_usage['shared_machine']:
            if self._gpu_usage['restrict_visible_devices']:
                os.environ['CUDA_VISIBLE_DEVICES'] = self._gpu_usage['CUDA_VISIBLE_DEVICES']

            if self._gpu_usage['allow_growth']:
                config.gpu_options.allow_growth = True

            if self._gpu_usage['restrict_per_process_gpu_memory_fraction']:
                config.gpu_options.per_process_gpu_memory_fraction = self._gpu_usage['per_process_gpu_memory_fraction']

        return config
