from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .model_analyser import ModelAnalyser



class BinaryModelAnalyser(ModelAnalyser):

    def plot_output_distribution(self,
                                 path_to_data,
                                 save_path,
                                 file_name
                                 ):


        predictions, labels, weights = self._get_predictions_labels_weights(path_to_val_data)

        df_predictions = pd.DataFrame(predictions, columns=['prediction'])
        df_labels      = pd.DataFrame(labels,      columns=['label'])
        df_weights     = pd.DataFrame(weights,     columns=['weight'])

        df = pd.concat([df_labels, df_predictions, df_weights], axis=1)


        signal_predictions = df.query('label==1')['prediction'].values
        signal_weights     = df.query('label==1')['weight'].values

        background_predictions = df.query('label==0')['prediction'].values
        background_weights     = df.query('label==0')['weight'].values


        bin_edges = np.linspace(0, 1, 30)

        plt.hist(signal_predictions,     bins=bin_edges, weights=signal_weights,     histtype='step', lw=1.5, label='Signal',     normed='True', color='#1f77b4')
        plt.hist(background_predictions, bins=bin_edges, weights=background_weights, histtype='step', lw=1.5, label='Background', normed='True', color='#d62728')

        plt.legend(loc='upper left')
        plt.xlabel('Network Output')
        plt.ylabel('Events (normalized)')

        plt.savefig(os.path.join(save_path, file_name + '.pdf'))
