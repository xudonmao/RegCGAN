from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import pdb
import random

FLAGS = tf.app.flags.FLAGS


class DATA():
  def __init__(self, batch_size, font_name):
    self._index_in_epoch = 0
    self._test_index = 0
    self._batch_size = batch_size
    self._font_name = font_name

  def load(self):
    X = np.load("{0}.npy".format(self._font_name))
    
    #X = np.concatenate((trX, teX), axis=0)
    #y = np.concatenate((trY, teY), axis=0)
    
    seed = random.randint(0, 1000000)
    np.random.seed(seed)
    np.random.shuffle(X)
    #np.random.seed(seed)
    #np.random.shuffle(y)
    
    #y_vec = np.zeros((len(y), FLAGS.y_dim), dtype=np.float)

    
    self._images = X/127.5 - 1.0
    self._num_examples = X.shape[0]

    
  def next_batch(self):
    start = self._index_in_epoch
    self._index_in_epoch += self._batch_size
    if self._index_in_epoch > self._num_examples:
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      #self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = self._batch_size
    end = self._index_in_epoch
    return self._images[start:end]

  def next_test_batch(self):
    start = self._test_index
    self._test_index += FLAGS.test_batch_size
    if self._test_index > self._test_num_examples:
      start = 0
      self._test_index = FLAGS.test_batch_size
    end = self._test_index
    return self._test_images[start:end], self._test_labels[start:end]
  

