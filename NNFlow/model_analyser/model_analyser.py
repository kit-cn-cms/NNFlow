from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd

import tensorflow as tf

from NNFlow.data_frame.data_frame import DataFrame


class ModelAnalyser(object):
    
    def __init__(self,
                 path_to_model,
                 number_of_output_neurons,
                 gpu_usage):


        self._path_to_model            = path_to_model
        self._number_of_output_neurons = number_of_output_neurons
        self._gpu_usage                = gpu_usage




    def save_variable_ranking(self,
                              save_dir,
                              path_to_variablelist):


        config = self._get_session_config()
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
                outfile.write('{:60} {}\n'.format(variable, str(variable_ranking[variable])))

        variable_ranking.to_msgpack(os.path.join(save_dir, 'variable_ranking.msg'))




    def _get_session_config(self):


        config = tf.ConfigProto()
        if self._gpu_usage['shared_machine']:
            if self._gpu_usage['restrict_visible_devices']:
                os.environ['CUDA_VISIBLE_DEVICES'] = self._gpu_usage['CUDA_VISIBLE_DEVICES']

            if self._gpu_usage['allow_growth']:
                config.gpu_options.allow_growth = True

            if self._gpu_usage['restrict_per_process_gpu_memory_fraction']:
                config.gpu_options.per_process_gpu_memory_fraction = self._gpu_usage['per_process_gpu_memory_fraction']


        return config




    def _get_labels_predictions_event_weights(self,
                                              path_to_input_file):


        data_set = DataFrame(path_to_input_file       = path_to_input_file,
                             number_of_output_neurons = self._number_of_output_neurons)
        
        data, labels, weights = data_labels_event_weights.get_data_labels_event_weights()


        config = self._get_session_config()
        graph = tf.Graph()
        with tf.Session(config=config, graph=graph) as sess:
            saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
            saver.restore(sess, path_to_model)

            input_data     = graph.get_tensor_by_name("input:0")
            network_output = graph.get_tensor_by_name("output:0")

            predictions = sess.run(network_output, {input_data : data})


        return labels, predictions, event_weights
