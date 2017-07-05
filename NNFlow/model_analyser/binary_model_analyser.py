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


        labels, predictions, event_weights = self._get_labels_predictions_event_weights(path_to_data)

        df_labels        = pd.DataFrame(labels,        columns=['label'])
        df_predictions   = pd.DataFrame(predictions,   columns=['prediction'])
        df_event_weights = pd.DataFrame(event_weights, columns=['event_weight'])

        df = pd.concat([df_labels, df_predictions, df_event_weights], axis=1)


        signal_predictions   = df.query('label==1')['prediction'].values
        signal_event_weights = df.query('label==1')['event_weight'].values

        background_predictions   = df.query('label==0')['prediction'].values
        background_event_weights = df.query('label==0')['event_weight'].values


        bin_edges = np.linspace(0, 1, 30)

        plt.hist(signal_predictions,     bins=bin_edges, weights=signal_event_weights,     histtype='step', lw=1.5, label='Signal',     normed='True', color='#1f77b4')
        plt.hist(background_predictions, bins=bin_edges, weights=background_event_weights, histtype='step', lw=1.5, label='Background', normed='True', color='#d62728')

        plt.legend(loc='upper left')
        plt.xlabel('Network Output')
        plt.ylabel('Events (normalized)')

        plt.savefig(os.path.join(save_path, file_name + '.pdf'))
