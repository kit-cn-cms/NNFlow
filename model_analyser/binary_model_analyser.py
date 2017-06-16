from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt

from model_analyser.model_analyser import ModelAnalyser



class BinaryModelAnalyser(ModelAnalyser):

    def print_output_distribution(self,
                                  path_to_train_data,
                                  path_to_val_data):


        predictions_train, labels_train, weights_train = self._get_predictions_labels_weights(path_to_train_data)
        predictions_val,   labels_val,   weights_val   = self._get_predictions_labels_weights(path_to_val_data)

        plt.hist()
