from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd

import tensorflow as tf

from NNFlow.data_frame.data_frame import DataFrame
from NNFlow.session_config.session_config import SessionConfig


class ModelAnalyser(object):
    
    def __init__(self,
                 path_to_model,
                 batch_size_classification = 200000,
                 session_config            = None,
                 ):


        self._path_to_model             = path_to_model
        self._batch_size_classification = batch_size_classification

        if session_config is not None:
            self._session_config = session_config
        else:
            self._session_config = SessionConfig()


        config = self._session_config.get_config()
        graph = tf.Graph()
        with tf.Session(config=config, graph=graph) as sess:
            saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
            saver.restore(sess, self._path_to_model)
            self._names_input_neurons = graph.get_tensor_by_name("names_input_neurons:0").eval()




    def save_variable_ranking(self,
                              save_dir,
                              ):


        config = self._session_config.get_config()
        graph = tf.Graph()
        with tf.Session(config=config, graph=graph) as sess:
             saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
             saver.restore(sess, self._path_to_model)
             weights = graph.get_tensor_by_name("W_1:0").eval()

        weight_abs = np.absolute(weights)
        weight_abs_mean = np.mean(weight_abs, axis=1)

        variable_ranking = pd.Series(weight_abs_mean, index=self._names_input_neurons)
        variable_ranking.sort_values(inplace=True)

        with open(os.path.join(save_dir, 'variable_ranking.txt'), 'w') as outfile:
            for variable in variable_ranking.index:
                outfile.write('{:60} {}\n'.format(variable, str(variable_ranking[variable])))

        variable_ranking.to_msgpack(os.path.join(save_dir, 'variable_ranking.msg'))




    def get_labels_network_output_event_weights(self,
                                                path_to_input_file):


        data_set = DataFrame(path_to_input_file = path_to_input_file,
                             network_type       = self._network_type)


        config = self._session_config.get_config()
        graph = tf.Graph()
        with tf.Session(config=config, graph=graph) as sess:
            saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
            saver.restore(sess, self._path_to_model)

            input_data                = graph.get_tensor_by_name("input:0")
            network_output_calculator = graph.get_tensor_by_name("output:0")


            batch_network_output_list = list()

            for batch_data, batch_labels, batch_event_weights in data_set.get_data_labels_event_weights_as_batches(batch_size                 = self._batch_size_classification,
                                                                                                                   sort_events_randomly       = False,
                                                                                                                   include_smaller_last_batch = True,
                                                                                                                   ):
                batch_network_output = sess.run(network_output_calculator, {input_data : batch_data})

                batch_network_output_list.append(batch_network_output)


            network_output        = np.concatenate(batch_network_output_list, axis=0)
            labels, event_weights = data_set.get_labels_event_weights()


        return labels, network_output, event_weights
