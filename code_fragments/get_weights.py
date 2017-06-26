from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import pandas as pd

sys.path.append('/storage/b/hilser/NNFlow')
from classify.classify import classify_test_sample_multinomial

path_to_model = '/storage/b/hilser/ttbb_analysis/neural_network/multinomial_bg-combined-with-ttcc_6jet3tag_without-cut/model/model.ckpt'
path_to_test_sample = '/storage/b/hilser/ttbb_analysis/neural_network/multinomial_bg-combined-with-ttcc_6jet3tag_without-cut/training_data/val.npy'
number_of_processes = 4

gpu_usage = dict()
gpu_usage['shared_machine']                           = True
gpu_usage['restrict_visible_devices']                 = True
gpu_usage['CUDA_VISIBLE_DEVICES']                     = '0' 
gpu_usage['allow_growth']                             = True
gpu_usage['restrict_per_process_gpu_memory_fraction'] = False
gpu_usage['per_process_gpu_memory_fraction']          = 0.1


predictions, true_values, weights, W = classify_test_sample_multinomial(path_to_model, path_to_test_sample, number_of_processes, gpu_usage)

bg_pred_bg_dist = np.zeros(100)
bg_pred_sig_dist = np.zeros(100)

for i in range(predictions.shape[0]):
    if np.argmax(predictions[i])!=0:
        bg_pred = predictions[i][0]
        bg_pred = np.int(np.floor(bg_pred * 100))

        if np.argmax(true_values[i]) == 0:
            bg_pred_bg_dist[bg_pred] += 1
        else:
            bg_pred_sig_dist[bg_pred] += 1

bg_pred_bg_dist *= 10000/bg_pred_bg_dist.sum()
bg_pred_sig_dist *= 10000/bg_pred_sig_dist.sum()

matrix = np.zeros((number_of_processes, number_of_processes))
for i in range(predictions.shape[0]):
    true_value = np.argmax(true_values[i])
    prediction = np.argmax(predictions[i])

    if predictions[i][0]>0.2:
        matrix[true_value][0] += weights[i]
    else:
        matrix[true_value][prediction] += weights[i]

true_over_false = matrix.diagonal()/(matrix.sum(axis=0)-matrix.diagonal())
signal_fraction = matrix.diagonal()/matrix.sum(axis=0)


for i in range(len(bg_pred_bg_dist)):
    print(i, 'bg:', bg_pred_bg_dist[i], 'sig:', bg_pred_sig_dist[i])

print(matrix)
print(true_over_false)
print(signal_fraction)
print(W.shape)
