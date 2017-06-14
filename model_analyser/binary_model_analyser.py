from __future__ import absolute_import, division, print_function

import tensorflow as tf

from model_analyser.model_analyser import ModelAnalyser



class BinaryModelAnalyser(ModelAnalyser):

    def print_output_distribution(self):


        gpu_config = self._get_gpu_config()
        graph = tf.Graph()
        with tf.Session(config=config, graph=graph) as sess:
            saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
            saver.restore(sess, path_to_model)

            x = graph.get_tensor_by_name("input:0")
            y = graph.get_tensor_by_name("output:0")

            Verschiebe das in allgemeine Funktion
