from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf


class ModelAnalyser(object):
    
    def __init__(self,
                 path_to_model,
                 gpu_usage):

        self._path_to_model = path_to_model
        self._gpu_usage = gpu_usage



    def variable_ranking(self):
