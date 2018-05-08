from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import random
import pdb

FLAGS = tf.app.flags.FLAGS


class DATA():
  def __init__(self, batch_size, test_batch_size):
    self._index_in_epoch = 0
    self._test_index = 0
    self._batch_size = batch_size
    self._test_batch_size = test_batch_size


  def load(self):
    data_dir = FLAGS.data_dir

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)
  
    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)
  
    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)
  
    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)
  
    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = trX
    y = trY
    
    seed = random.randint(0, 100000)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i,int(y[i])] = 1.0

    
    X = X[0:2000,:]
    y_vec = y_vec[0:2000,:]
    self._images = X/255.
    self._labels = y_vec
    self._num_examples = y_vec.shape[0]

    y_vec_test = np.zeros((len(teY), 10), dtype=np.float)
    for i, label in enumerate(teY):
        y_vec_test[i,int(teY[i])] = 1.0
    self._test_images = teX/255.
    self._test_labels = y_vec_test
    self._test_num_examples = y_vec_test.shape[0]


    
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
    self._test_index += self._test_batch_size
    if self._test_index > self._test_num_examples:
      start = 0
      self._test_index = self._test_batch_size
    end = self._test_index
    return self._test_images[start:end], self._test_labels[start:end]
  

