from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import regcgan as gan
import os
import pdb
import usps as data1
import mnist as data2
from utils import *


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('start_regD', '2000', """learning rate""")
tf.app.flags.DEFINE_float('weight_D', '0.004', """learning rate""")
tf.app.flags.DEFINE_float('weight_G', '0.1', """learning rate""")
tf.app.flags.DEFINE_float('learning_rate', '0.0002', """learning rate""")
tf.app.flags.DEFINE_float('beta1', '0.5', """beta for Adam""")
tf.app.flags.DEFINE_integer('batch_size', '64', """batch size""")
tf.app.flags.DEFINE_integer('half_batch_size', '32', """batch size""")
tf.app.flags.DEFINE_integer('domain_num', '2', """y dim""")
tf.app.flags.DEFINE_integer('y_dim', '10', """y dim""")
tf.app.flags.DEFINE_integer('c_dim', '1', """y dim""")
tf.app.flags.DEFINE_integer('map_num', '64', """y dim""")
tf.app.flags.DEFINE_integer('output_size', '28', """output size""")
tf.app.flags.DEFINE_integer('image_size', '28', """output size""")
tf.app.flags.DEFINE_integer('test_sample_size', '2007', """sample number""")
tf.app.flags.DEFINE_integer('test_batch_size', '223', """batch size""")
tf.app.flags.DEFINE_integer('test_sample_size2', '10000', """sample number""")
tf.app.flags.DEFINE_integer('test_batch_size2', '100', """batch size""")

tf.app.flags.DEFINE_string('data_dir', './data/', """data dir""")
tf.app.flags.DEFINE_integer('max_steps', 150001,
                                """Number of batches to run.""")


def sess_init():
  init = tf.global_variables_initializer()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(init)
  return sess

def load_data():
  data_set = data1.DATA(FLAGS.half_batch_size, FLAGS.test_batch_size)
  data_set.load()
  data_set2 = data2.DATA(FLAGS.half_batch_size, FLAGS.test_batch_size2)
  data_set2.load()
  return data_set, data_set2

def GetVars():
  t_vars = tf.trainable_variables()
  G_vars = [var for var in t_vars if 'g_' in var.name]
  D_vars = [var for var in t_vars if 'd_' in var.name]
  C_vars = [var for var in t_vars if 'cls_' in var.name]
  return G_vars, D_vars, C_vars

def Holder():
  z_h = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, gan.Z_DIM])
  images_h = tf.placeholder(tf.float32, 
      shape=[FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim])
  labels_h1 = tf.placeholder(tf.float32, shape=[FLAGS.half_batch_size, FLAGS.y_dim])
  domain_labels_h = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.domain_num])
  return z_h, images_h, labels_h1, domain_labels_h

def holder_test():
  test_images_h = tf.placeholder(tf.float32, 
      shape=[FLAGS.test_batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim])
  test_labels_h = tf.placeholder(tf.float32, shape=[FLAGS.test_batch_size, FLAGS.y_dim])

  test_images_h2 = tf.placeholder(tf.float32, 
      shape=[FLAGS.test_batch_size2, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim])
  test_labels_h2 = tf.placeholder(tf.float32, shape=[FLAGS.test_batch_size2, FLAGS.y_dim])

  test_domain_labels_h = tf.placeholder(tf.float32, shape=[FLAGS.test_batch_size, FLAGS.domain_num])
  test_domain_labels_h2 = tf.placeholder(tf.float32, shape=[FLAGS.test_batch_size2, FLAGS.domain_num])

  return test_images_h, test_images_h2, test_labels_h, test_labels_h2, test_domain_labels_h, test_domain_labels_h2



def GenValsForHolder(data_set, data_set2):
  z_v = np.random.uniform(-1, 1, 
      [FLAGS.batch_size, gan.Z_DIM]).astype(np.float32)

  z_v_half = np.random.uniform(-1, 1, 
      [FLAGS.half_batch_size, gan.Z_DIM]).astype(np.float32)
  z_for_G_v = np.concatenate((z_v_half, z_v_half),0)

  images_v1, labels_v1 = data_set.next_batch()
  images_v2, labels_v2 = data_set2.next_batch()
  images_v = np.concatenate((images_v1[0:FLAGS.half_batch_size,:], images_v2[0:FLAGS.half_batch_size,:]), 0)
  domain_labels_v1 = np.zeros((FLAGS.half_batch_size, 2))
  domain_labels_v1[:,0] = 1
  domain_labels_v2 = np.zeros((FLAGS.half_batch_size, 2))
  domain_labels_v2[:,1] = 1
  domain_labels_v = np.concatenate((domain_labels_v1, domain_labels_v2), 0)
  return z_v, z_for_G_v, images_v, labels_v1, domain_labels_v


def Eval(para_list, sess, step, message):
  test_op, test_images_h, test_labels_h, test_domain_labels_h,\
      test_domain_labels_v, data_set, test_batch_size, test_sample_size = para_list
  correct_num_count = 0
  iter_num = test_sample_size // test_batch_size

  for i in range(iter_num):
    test_images_v, test_labels_v = data_set.next_test_batch()
    correct_num_count += sess.run(test_op, 
        feed_dict={test_images_h:test_images_v, test_labels_h:test_labels_v,\
            test_domain_labels_h:test_domain_labels_v})

  print "%s step = %d, accuracy = %f" % (message, step, correct_num_count / \
      (test_batch_size*iter_num))


def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    #input
    z_h, images_h, labels_h1, domain_labels_h = Holder()
    test_images_h1, test_images_h2, test_labels_h1, test_labels_h2, \
        test_domain_labels_h, test_domain_labels_h2 = holder_test()
    reg_D_weight_h = tf.placeholder(tf.float32)
    reg_G_weight_h = tf.placeholder(tf.float32)

    
    #inference
    D_logits_real, D_logits_fake, D_logits_fake_for_G, reg_g, reg_d = \
        gan.inference(images_h, domain_labels_h, z_h)

    C_logits = gan.classifier(images_h, domain_labels_h, FLAGS.batch_size)

    sampler = gan.sampler(z_h, domain_labels_h)

    #loss
    G_loss, D_loss = gan.loss(D_logits_real, D_logits_fake, D_logits_fake_for_G, \
                       reg_g, reg_d, reg_G_weight_h, reg_D_weight_h, C_logits, labels_h1)


    #train_op
    G_vars, D_vars, C_vars = GetVars()
    G_train_op, D_train_op = gan.train(G_loss, D_loss, G_vars, D_vars, global_step)


    #for_eval
    test_logits = gan.classifier(test_images_h1, test_domain_labels_h, FLAGS.test_batch_size, True)
    test_op = gan.correct_num(test_logits, test_labels_h1)
    test_logits2 = gan.classifier(test_images_h2, test_domain_labels_h2, FLAGS.test_batch_size2, True)
    test_op2 = gan.correct_num(test_logits2, test_labels_h2)

    data_set, data_set2 = load_data()


    sess = sess_init()
    saver = tf.train.Saver()

    for step in xrange(FLAGS.max_steps):

      _, z_for_G_v, images_v, labels_v1, domain_labels_v = \
          GenValsForHolder(data_set, data_set2)

      errC, errD, errG, reg_g_v, reg_d_v = (0,0,0,0,0)

      if step > FLAGS.start_regD:
          _, errD, reg_d_v = sess.run([D_train_op, D_loss, reg_d],
              feed_dict={z_h:z_for_G_v, images_h:images_v, labels_h1:labels_v1,\
                  domain_labels_h:domain_labels_v, reg_D_weight_h:FLAGS.weight_D})
      else:
          _, errD, reg_d_v = sess.run([D_train_op, D_loss, reg_d],
              feed_dict={z_h:z_for_G_v, images_h:images_v, labels_h1:labels_v1,\
                  domain_labels_h:domain_labels_v, reg_D_weight_h:0.0})

      _, errG, reg_g_v = sess.run([G_train_op, G_loss, reg_g],
          feed_dict={z_h:z_for_G_v, domain_labels_h:domain_labels_v, \
              reg_G_weight_h:FLAGS.weight_G})


      if step % 100 == 0:
        print "step = %d, errD = %f, errG = %f errC = %f reg_g = %f reg_d = %f" \
            % (step, errD, errG, errC, reg_g_v, reg_d_v)


      if step % 1000 == 0:
        test_domain_labels_v = np.zeros((FLAGS.test_batch_size, 2))
        test_domain_labels_v[:,0] = 1
        test_domain_labels_v2 = np.zeros((FLAGS.test_batch_size2, 2))
        test_domain_labels_v2[:,1] = 1
        para_list = (test_op, test_images_h1, test_labels_h1, test_domain_labels_h, \
            test_domain_labels_v, data_set, FLAGS.test_batch_size, FLAGS.test_sample_size)
        Eval(para_list, sess, step, 'Source')
        para_list = (test_op2, test_images_h2, test_labels_h2, test_domain_labels_h2, \
            test_domain_labels_v2, data_set2, FLAGS.test_batch_size2, FLAGS.test_sample_size2)
        Eval(para_list, sess, step, 'Target')


        z_v_half = np.random.uniform(-1, 1, 
            [FLAGS.half_batch_size, gan.Z_DIM]).astype(np.float32)
        z_v = np.concatenate((z_v_half, z_v_half),0)

        domain_labels_v = np.zeros((FLAGS.batch_size, 2))
        domain_labels_v[0:FLAGS.half_batch_size, 0] = 1
        domain_labels_v[FLAGS.half_batch_size:, 1] = 1

        samples = sess.run(sampler, 
            feed_dict={z_h:z_v, domain_labels_h:domain_labels_v})
        save_images(samples, [2, FLAGS.half_batch_size],
            './samples/train_{:d}.bmp'.format(step))


def main(argv=None):
  train()

if __name__ == "__main__":
  tf.app.run()
