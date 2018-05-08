from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import pdb
from utils import * 

FLAGS = tf.app.flags.FLAGS

class Data():
  def __init__(self):
	pass

  def read_and_decode(self, filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
  
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([120*160*3])
    image = tf.reshape(image, [120,160,3])
  
    image = tf.cast(image, tf.float32) * (1. / 127.5) - 1.0
  
    return image
  
  
  def load(self, filename, batch_size):
  
    filename_queue = tf.train.string_input_producer(
        [filename])
  
    image = self.read_and_decode(filename_queue)
  
    images = tf.train.shuffle_batch(
        [image], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)
  
    return images

