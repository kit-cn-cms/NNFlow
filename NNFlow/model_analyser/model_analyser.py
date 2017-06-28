from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd

import tensorflow as tf

from data_frame.data_frame import DataFrame


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




    def _get_predictions_labels_weights(self,
                                        path_to_input_file):


        data_labels_weights = DataFrame(path_to_input_file       = path_to_input_file,
                                        number_of_output_neurons = self._number_of_output_neurons)
        
        data, labels, weights = data_labels_weights.get_data_labels_weights()


        config = self._get_session_config()
        graph = tf.Graph()
        with tf.Session(config=config, graph=graph) as sess:
            saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
            saver.restore(sess, path_to_model)

            x = graph.get_tensor_by_name("input:0")
            y = graph.get_tensor_by_name("output:0")

            feed_dict = {x:data}
            predictions = sess.run(y, feed_dict)


        return predictions, labels, weights
