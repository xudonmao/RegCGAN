from __future__ import absolute_import
from __future__ import division

import os
import pdb
import tensorflow as tf
import numpy as np
import random
import scipy.misc
import sys



tf.app.flags.DEFINE_string('directory', './',
                           'Directory to download data files and write the '
                           'converted result')
tf.app.flags.DEFINE_integer('validation_size', 5000,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
FLAGS = tf.app.flags.FLAGS

def write_txt(f, x, y):
  s = x.shape
  for a in range(s[0]):
    for b in range(s[1]):
      for c in range(s[2]):
        for d in range(s[3]):
          f.write('{0} '.format(x[a,b,c,d]))
    f.write('{0}\n'.format(y))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)

  #f = open('before.txt', 'w+')

  for label, npy in enumerate(os.listdir(data_set)):
    npy_path = os.path.join(data_set, npy)
    X = np.load(npy_path)

    num_examples = X.shape[0]
    rows = X.shape[1]
    cols = X.shape[2]
    depth = X.shape[3]
    #write_txt(f, X, label)

    for i in range(num_examples):
      image_raw = X[i,:].tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
          'height': _int64_feature(rows),
          'width': _int64_feature(cols),
          'depth': _int64_feature(depth),
          'label': _int64_feature(int(label)),
          'image_raw': _bytes_feature(image_raw)}))
      writer.write(example.SerializeToString())
  #f.close()
  writer.close()

def img_to_tf(file_list, name):
  #pdb.set_trace()
  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  writer = tf.python_io.TFRecordWriter(filename)
  
  with open(file_list) as f:
    file_set = [line.strip('\n') for line in f]
  random.shuffle(file_set)

  output_size = 64

  for i, line in enumerate(file_set):
    img_path = line
    img = scipy.misc.imread(img_path, flatten=True)
    img = scipy.misc.imresize(img, [output_size, output_size])

    rows = img.shape[0]
    cols = img.shape[1]
    depth = 1

    image_raw = img.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    if i % 10000 == 0:
      print i

  writer.close()



def main(argv):
  # Get the data.
  #img_to_tf('./file.list.test', 'icdar28_test')
  img_to_tf(sys.argv[1], sys.argv[2])

  # Convert to Examples and write the result to TFRecords.
  #convert_to("./icdar_small_10/", 'icdar20')
  #convert_to("../icdar28_test/", 'icdar_test')


if __name__ == '__main__':
  tf.app.run()
