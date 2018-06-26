# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
# Import some stuff
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# dynamically allocate GPU RAM thereby not blocking GPU for other users
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def main(_):
  # Import data
  #  mnist = input_data.read_data_sets(FLAGS.data_dir)
  mnist = input_data.read_data_sets("/local/scratch/NNWorkshop/datasets/MNIST_data/")

  # Create the model

  # Here we define the tensors, their type and dimension
  x = tf.placeholder(tf.float32, [None, 784]) # input feature vector with dimension 784 corresponding to the pixels of the mnist pictures
  W = tf.Variable(tf.zeros([784, 10])) # We want to have 10 neurons in the hidden layer. Each neuron should be connected to all 784 features of the input vector. -> 784x10 weights. Intialize the weights to zero
  b = tf.Variable(tf.zeros([10])) # bias weights. One for each neuron.

  # define a mathematical operation. In tensorflow you define the operations at the beginning and then later run the defined operations
  # Here y is the output value of the neural network given the input  feature vector x
  # So this operation defines how the output value should be calculated from the input vector and the trainable weights W and biases b
  # Question: Which dimension does y have (phrased differently: How many output nodes has the NN?)
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None]) # placeholder vector for true labels of pictures

  # As loss function we will use the cross-entropy
  # We also add a softmax laxer at the output of the NN. With this we get a 1 at the output node with the highest node value and 0 at the other nodes. Why?
  # But the raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
  # outputs of 'y', and then average across the batch.
  # Define
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  # here we define which optimization algorithm we want to use and which function to minimize (cross_entropy)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  # In principle we could define arbitrary operations. For example just simple matrix multiplication.
  # Here we have a training just because we used the GradientDescentOptimizer Method in the train_step
  # You can think of the definiton of the tensorflow Graph as just one large complicated formula

  # Now we define the session. A class to manage the training.
  sess = tf.InteractiveSession()
  # initialize all the variables
  tf.global_variables_initializer().run()
  # Train 
  # We will do 1000 training steps with 100 pictures per step
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # Each step we tell tensorflow to run the operation defined as train_step with the given input and label vectors
    # In principle we could run arbitrary operations. For example just simple matrix multiplication.
    # Here we have a training just because we used the GradientDescentOptimizer Method in the train_step

  # Test trained model
  # To quantify how well our NN worked we evaluate the NN on previously unseen pictures and check if the predicted class corresponds to the true class of the picture
  # We calculate the fraction of pictures for which this is true.
  # NB: The reduce_mean function calculates the mean of all values in the given tensor. Here add either 1 (if prediction is correct) or 0 (if prediction is wrong) and then divide by the total number of classified pictures
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  print("Final ROC integral on evaluation data")
  # run the defined accuracy operation with the given input tensors
  print(sess.run(
      accuracy, feed_dict={
          x: mnist.test.images,
          y_: mnist.test.labels
      }))

# general main function. Not important for us
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='mnist/input_data',
      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

