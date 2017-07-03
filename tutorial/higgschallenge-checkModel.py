# Marco A. Harrendorf, Lukas Hilser

import numpy as np
import tensorflow as tf


def main():
  # GPU usage requirements
  gpu_usage = dict()
  gpu_usage['shared_machine']                           = True
  gpu_usage['restrict_visible_devices']                 = False
  gpu_usage['CUDA_VISIBLE_DEVICES']                     = '0'
  gpu_usage['allow_growth']                             = True
  gpu_usage['restrict_per_process_gpu_memory_fraction'] = False
  gpu_usage['per_process_gpu_memory_fraction'] = 0.1


  modelAnalyser =  ModelAnalyser('Higgschallenge/Higgschallenge.ckpt','/local/scratch/NNWorkshop/datasets/higgschallenge/val.npy',3,gpu_usage)
  print "Category 0: ttH, Category 1: ttbb, Category 2: ttlf"
  correctAssignment, correctCategoryAssignment = modelAnalyser.getValidationOutput()
  print correctCategoryAssignment


### Begin class definition 
class ModelAnalyser(object):
  """Class using tensorflow model and giving accurateness on provided input data set"""

    
  def __init__(self,
                path_to_model, 
                path_to_input_file,
                number_of_output_neurons,
                gpu_usage):
    """ Default constructor
    path_to_model: Determines Tensorflow model file location. Note: Provide path and filename without ".meta" ending
    path_to_input_file: Determines location of input data set file
    number_of_output_neurons: Defines how many neurons are used in the output layer
    gpu_usage: Define gpu options
    """
    # Init member variables
    self._path_to_model            = path_to_model
    self._path_to_input_file       = path_to_input_file
    self._number_of_output_neurons = int(number_of_output_neurons)
    self._gpu_usage                = gpu_usage
    self._predictions              = None
    self._labels                   = None
    self._weights                  = None
    self._graph                    = tf.Graph()
    
    # Init member variables containing validation output
    self._correctAssignment = None
    self._correctCategoryAssignment = None
    
    self._getPredictions()
    self._validate()

  def _getSessionGpuConfig(self):
    """Make sure gpu options are properly set"""
    config = tf.ConfigProto()
    if 'shared_machine' in self._gpu_usage and self._gpu_usage['shared_machine']:
        if 'restrict_visible_devices' in self._gpu_usage and self._gpu_usage['restrict_visible_devices'] and 'CUDA_VISIBLE_DEVICES' in self._gpu_usage:
            os.environ['CUDA_VISIBLE_DEVICES'] = self._gpu_usage['CUDA_VISIBLE_DEVICES']

        if 'allow_growth' in self._gpu_usage and self._gpu_usage['allow_growth']:
            config.gpu_options.allow_growth = True

        if 'restrict_per_process_gpu_memory_fraction' in self._gpu_usage and self._gpu_usage['restrict_per_process_gpu_memory_fraction']:
            config.gpu_options.per_process_gpu_memory_fraction = self._gpu_usage['per_process_gpu_memory_fraction']
    return config


  def _getPredictions(self):
    """Function returns the predictions based on an existing Tensorflow model"""
    # Load data
    dataFrame = np.load(self._path_to_input_file)
    data = dataFrame[:, self._number_of_output_neurons:-1]
    self._weights = dataFrame[:, -1]
    if self._number_of_output_neurons == 1:
      self._labels = dataFrame[:, 0]
    else:
      self._labels = dataFrame[:, :self._number_of_output_neurons]

    with tf.Session(config=self._getSessionGpuConfig(),graph=self._graph) as sess:
        saver = tf.train.import_meta_graph(self._path_to_model+ '.meta')
        saver.restore(sess, self._path_to_model)

        input_data  = self._graph.get_tensor_by_name("input:0")
        network_output = self._graph.get_tensor_by_name("output:0")

        self._predictions = sess.run(network_output, {input_data : data})
      
  def _validate(self):
    """Evaluates the training process.  """
    # Init arrays containing either only truely assigned events and all predicted events  
    arr_cross = np.zeros((self._number_of_output_neurons, self._number_of_output_neurons),dtype=np.float32)
    index_true = np.argmax(self._labels, axis=1)
    index_pred = np.argmax(self._predictions, axis=1)
    
    # Determine overall accurateness
    for i in range(index_true.shape[0]):
      arr_cross[index_true[i]][index_pred[i]] += self._weights[i]
    correctAssignment = np.diagonal(arr_cross).sum()
    wrongAssignment = arr_cross.sum() - correctAssignment
    print "Overall correct assignment: ",  correctAssignment, "Wrong assignment: ", wrongAssignment
    self._correctAssignment = correctAssignment
    
    # Determine category accurateness
    correctCategoryAssignment = np.zeros((self._number_of_output_neurons), dtype=np.float32)
    for i in range(self._number_of_output_neurons): 
      correctCategoryAssignment[i] = arr_cross[i][i] / (np.sum(arr_cross, axis=1)[i])
      print "Percentage of correctly assigned events in category ", i, ": ", correctCategoryAssignment[i]
    self._correctCategoryAssignment = correctCategoryAssignment

  def getValidationOutput(self):
    return self._correctAssignment, self._correctCategoryAssignment

### End class definition          
      
    



if __name__ == "__main__":
  main()


 
