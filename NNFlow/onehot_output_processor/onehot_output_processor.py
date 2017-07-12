from __future__ import absolute_import, division, print_function

import numpy as np




class OneHotOutputProcessor(object):


    def __init__(self):


        self.set_mode_max_output()




    def get_predicted_true_matrix(self,
                                  labels,
                                  network_output,
                                  event_weights,
                                  cross_sections = None,
                                  ):


        number_of_events         = labels.shape[0]
        number_of_output_neurons = labels.shape[1]

        array_predicted_true = np.zeros((number_of_output_neurons, number_of_output_neurons), dtype=np.float32)

        for i in range(number_of_events):
            index_true      = np.argmax(labels[i])
            index_predicted = self.get_prediction(network_output[i])

            array_predicted_true[index_true][index_predicted] += event_weights[i]


        if cross_sections is not None:
            for j in range(number_of_output_neurons):
                array_predicted_true[j] /= array_predicted_true[j].sum()

            if cross_sections == 'equal':
                array_predicted_true *= number_of_output_neurons
                array_predicted_true *= 100

            else:
                for j in range(number_of_output_neurons):
                    array_predicted_true[j] *= cross_sections[j]


        return array_predicted_true




    def get_prediction(self,
                       network_output,
                       ):


        if self._mode == 'max_output':
            prediction = np.argmax(network_output)


        elif self._mode == 'min_output_to_accept':

            if np.argmax(network_output) == self._min_output_to_accept_index:
                if network_output[self._min_output_to_accept_index] > self._min_output_to_accept_value
                    prediction = self._min_output_to_accept_index

                else:
                    prediction = np.argsort(network_output)[-2]

            else:
                prediction = np.argmax(network_output)


        elif self._mode == 'max_output_to_reject':

            if network_output[self._max_output_to_reject_index] > self._max_output_to_reject_value:
                prediction = self._max_output_to_reject_index

            else:
                prediction = np.argmax(network_output)


        return prediction




    def set_mode_max_output(self):

        self._mode = 'max_output'




    def set_mode_min_output_to_accept(self,
                                      index,
                                      value,
                                      ):


        self._mode = 'min_output_to_accept'

        self._min_output_to_accept_index = process
        self._min_output_to_accept_value = value




    def set_mode_min_output_to_reject(self,
                                      index,
                                      value,
                                      ):


        self._mode = 'max_output_to_reject'

        self._max_output_to_reject_index = index
        self._max_output_to_reject_value = value
