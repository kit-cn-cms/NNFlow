from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd

import tensorflow as tf

from NNFlow.data_frame.data_frame         import DataFrame     as NNFlowDataFrame
from NNFlow.session_config.session_config import SessionConfig as NNFlowSessionConfig




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
            self._session_config = NNFlowSessionConfig()


        tf_config = self._session_config.get_tf_config()
        tf_graph  = tf.Graph()
        with tf.Session(config=tf_config, graph=tf_graph) as tf_session:
            tf_saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
            tf_saver.restore(tf_session, self._path_to_model)
            self._input_variables = tf_graph.get_tensor_by_name("input_variables:0").eval()


        self._labels_network_output_event_weights = dict()




    def get_labels_network_output_event_weights(self,
                                                path_to_input_file,
                                                ):


        if path_to_input_file in self._labels_network_output_event_weights.keys():
            return self._labels_network_output_event_weights[path_to_input_file]


        data_set = NNFlowDataFrame(path_to_input_file)


        tf_config = self._session_config.get_tf_config()
        tf_graph  = tf.Graph()
        with tf.Session(config=tf_config, graph=tf_graph) as tf_session:
            tf_saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
            tf_saver.restore(tf_session, self._path_to_model)

            tf_input_data     = tf_graph.get_tensor_by_name("input:0")
            tf_network_output = tf_graph.get_tensor_by_name("output:0")


            batch_network_output_list = list()

            for batch_data, batch_labels, batch_event_weights in data_set.get_data_labels_event_weights_as_batches(batch_size                 = self._batch_size_classification,
                                                                                                                   sort_events_randomly       = False,
                                                                                                                   include_smaller_last_batch = True,
                                                                                                                   ):
                batch_network_output = tf_session.run(tf_network_output, {tf_input_data : batch_data})

                batch_network_output_list.append(batch_network_output)


            network_output        = np.concatenate(batch_network_output_list, axis=0)
            labels, event_weights = data_set.get_labels_event_weights()


        self._labels_network_output_event_weights[path_to_input_file] = (labels, network_output, event_weights)


        return labels, network_output, event_weights




    def save_unit_test_data(self,
                            path_to_input_file,
                            save_dir,
                            ):


        data_set = NNFlowDataFrame(path_to_input_file = path_to_input_file)

        data = data_set.get_data()[0]

        network_output = self.get_labels_network_output_event_weights(path_to_input_file)[1][0]


        with open(os.path.join(save_dir, 'unitTestInputValues.txt'), 'w') as file_input_values:
            for input_value in data:
                file_input_values.write(str(input_value) + '\n')


        with open(os.path.join(save_dir, 'unitTestOutputValues.txt'), 'w') as file_output_values:
            if self._network_type == 'binary':
                file_output_values.write(str(network_output) + '\n')

            else:
                for output_value in network_output:
                    file_output_values.write(str(output_value) + '\n')




    def save_variable_ranking(self,
                              save_dir,
                              ):


        tf_config = self._session_config.get_tf_config()
        tf_graph  = tf.Graph()
        with tf.Session(config=tf_config, graph=tf_graph) as tf_session:
             tf_saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
             tf_saver.restore(tf_session, self._path_to_model)
             weights = tf_graph.get_tensor_by_name("W_1:0").eval()

        weight_abs = np.absolute(weights)
        weight_abs_mean = np.mean(weight_abs, axis=1)

        variable_ranking = pd.Series(weight_abs_mean, index=self._input_variables)
        variable_ranking.sort_values(inplace=True)

        with open(os.path.join(save_dir, 'variable_ranking.txt'), 'w') as outfile:
            for variable in variable_ranking.index:
                outfile.write('{:60} {}\n'.format(variable, str(variable_ranking[variable])))

        variable_ranking.to_msgpack(os.path.join(save_dir, 'variable_ranking.msg'))
