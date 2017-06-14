from __future__ import absolute_import, division, print_function

import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_auc_score


def classify_test_sample_binary(path_to_model, path_to_test_sample, gpu_usage):

    array_test_sample = np.load(path_to_test_sample)

    network_input = array_test_sample[:, 1:-1]
    true_values = array_test_sample[:, :1]


    config = tf.ConfigProto()
    if gpu_usage['shared_machine']:
        if gpu_usage['restrict_visible_devices']:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_usage['CUDA_VISIBLE_DEVICES']

        if gpu_usage['allow_growth']:
            config.gpu_options.allow_growth = True

        if gpu_usage['restrict_per_process_gpu_memory_fraction']:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_usage['per_process_gpu_memory_fraction']


    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(path_to_model + '.meta')
        saver.restore(sess, path_to_model)

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("input:0")

        feed_dict ={x:network_input}

        y = graph.get_tensor_by_name("output:0")
        prediction = sess.run(y, feed_dict)

    
    return prediction, true_values, roc_auc_score(np.concatenate(true_values), np.concatenate(prediction))




def classify_test_sample_multinomial(path_to_model, path_to_test_sample, number_of_processes, gpu_usage):

    array_test_sample = np.load(path_to_test_sample)

    network_input = array_test_sample[:, number_of_processes:-1]
    weights = array_test_sample[:, -1]
    true_values = array_test_sample[:, :number_of_processes]


    config = tf.ConfigProto()
    if gpu_usage['shared_machine']:
        if gpu_usage['restrict_visible_devices']:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_usage['CUDA_VISIBLE_DEVICES']

        if gpu_usage['allow_growth']:
            config.gpu_options.allow_growth = True

        if gpu_usage['restrict_per_process_gpu_memory_fraction']:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_usage['per_process_gpu_memory_fraction']


    graph = tf.Graph()
    with tf.Session(config=config, graph=graph) as sess:
        saver = tf.train.import_meta_graph(path_to_model + '.meta')
        saver.restore(sess, path_to_model)
            
        x = graph.get_tensor_by_name("input:0")

        feed_dict ={x:network_input}

        y = graph.get_tensor_by_name("output:0")
        predictions = sess.run(y, feed_dict)

        W = graph.get_tensor_by_name("W_1:0").eval()

    
    return predictions, true_values, weights, W
