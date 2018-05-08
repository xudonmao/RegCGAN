from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import pdb
import random
from scipy import misc

FLAGS = tf.app.flags.FLAGS


class DATA():
  def __init__(self, batch_size):
    self._index_in_epoch = 0
    self._test_index = 0
    self._batch_size = batch_size


  def load(self):
    data_dir = FLAGS.data_dir

    X = np.load(os.path.join(data_dir, 'zip_train_X.npy'))
    y = np.load(os.path.join(data_dir, 'zip_train_Y.npy'))
    #X = X[0:1800,:]
    #y = y[0:1800]
    
    #seed = 547
    seed = random.randint(0, 100000)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i,int(y[i])] = 1.0

    
    self._images = X/255.
    self._labels = y_vec
    self._num_examples = y_vec.shape[0]

    
  def next_batch(self):
    start = self._index_in_epoch
    self._index_in_epoch += self._batch_size
    if self._index_in_epoch > self._num_examples:
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = self._batch_size
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

  def next_test_batch(self):
    start = self._test_index
    self._test_index += FLAGS.test_batch_size
    if self._test_index > self._test_num_examples:
      start = 0
      self._test_index = FLAGS.test_batch_size
    end = self._test_index
    return self._test_images[start:end], self._test_labels[start:end]
  

  def convert(self):
    data_dir = FLAGS.data_dir

    loaded = np.loadtxt(os.path.join(data_dir, 'zip.train'))
    X_small = loaded[:,1:].reshape(7291, 16,16)
    Y = loaded[:,1]

    X = np.zeros((7291, 28, 28))    
    for i in range(7291):
      X[i,:] = misc.imresize(X_small[i,:].reshape(16,16), (28,28))

    X = X.reshape((7291,28,28,1))
    np.save('../zip_train_X', X)
    np.save('../zip_train_Y', Y)

    loaded = np.loadtxt(os.path.join(data_dir, 'zip.test'))
    X_small = loaded[:,1:].reshape(2007, 16,16)
    Y = loaded[:,1]

    X = np.zeros((2007, 28, 28))    
    for i in range(2007):
      X[i,:] = misc.imresize(X_small[i,:].reshape(16,16), (28,28))

    X = X.reshape((2007,28,28,1))
    np.save('../zip_test_X', X)
    np.save('../zip_test_Y', Y)
