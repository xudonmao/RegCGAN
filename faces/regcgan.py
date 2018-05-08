from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import pdb

FLAGS = tf.app.flags.FLAGS
from ops import *
from utils import *


Z_DIM = 100
#FLAGS.domain_num = 4

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')


def discriminator(image, domain_y, reuse=False, for_G=False):
  with tf.variable_scope('discriminator'): 
    if reuse:
      tf.get_variable_scope().reuse_variables()

    domain_y_conv = tf.reshape(domain_y, [FLAGS.batch_size, 1, 1, FLAGS.domain_num])

    x = conv_cond_concat(image, domain_y_conv)
    #domain_y_conv = tf.ones((FLAGS.batch_size, FLAGS.domain_num, FLAGS.output_size, \
    #    FLAGS.c_dim)) * tf.reshape(domain_y, [FLAGS.batch_size, FLAGS.domain_num,1,1])
    #x = tf.concat((domain_y_conv, image), axis=1)
    #x = tf.concat((domain_y_conv, x), axis=1)

    h0 = lrelu(conv2d(x, 64, name='d_h0_conv'), name='d_h0_relu')
    h1 = lrelu(d_bn1(conv2d(h0, 128, name='d_h1_conv')), name='d_h1_relu')
    h2 = lrelu(d_bn2(conv2d(h1, 256, name='d_h2_conv')), name='d_h2_relu')
    h3 = lrelu(d_bn3(conv2d(h2, 512, name='d_h3_conv')), name='d_h3_relu')
    
    h3_source = h3[0:FLAGS.half_batch_size, :]
    h3_target = h3[FLAGS.half_batch_size:, :]
    reg_d = tf.reduce_mean(tf.nn.l2_loss(h3_source - h3_target))

    h4 = linear(tf.reshape(h3, [FLAGS.batch_size, -1]), 1, 'd_h4_lin')

    return h4, reg_d

def generator(z, domain_y):
  with tf.variable_scope('generator'):
    s = FLAGS.output_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    z = tf.concat(axis=1, values=[z, domain_y])
    domain_y_conv = tf.reshape(domain_y, [FLAGS.batch_size, 1, 1, FLAGS.domain_num])

    h0 = tf.nn.relu(g_bn0(linear(z, 64*8*s16*s16, 'g_h0_lin')), name='g_h0_relu')

    h0_source = h0[0:FLAGS.half_batch_size, :]
    h0_target = h0[FLAGS.half_batch_size:, :]
    reg_g = tf.reduce_mean(tf.nn.l2_loss(h0_source - h0_target))

    h0 = tf.reshape(h0, [FLAGS.batch_size, s16, s16, 512])
    h0 = conv_cond_concat(h0, domain_y_conv)

    h1 = tf.nn.relu(g_bn1(deconv2d(h0,[FLAGS.batch_size, s8, s8, 256], name='g_h1')),name='g_h1_relu')
    h1 = conv_cond_concat(h1, domain_y_conv)

    h2 = tf.nn.relu(g_bn2(deconv2d(h1,[FLAGS.batch_size, s4, s4, 128], name='g_h2')),name='g_h2_relu')
    h2 = conv_cond_concat(h2, domain_y_conv)

    h3 = tf.nn.relu(g_bn3(deconv2d(h2,[FLAGS.batch_size, s2, s2, 64], name='g_h3')),name='g_h3_relu')
    h3 = conv_cond_concat(h3, domain_y_conv)

    h4 = tf.nn.tanh(deconv2d(h3, [FLAGS.batch_size, s, s, 3], name='g_h4'), name='g_h4_sigmoid')

    return h4, reg_g

def sampler(z, domain_y):
  with tf.variable_scope('generator'):
    tf.get_variable_scope().reuse_variables()


    s = FLAGS.output_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    z = tf.concat(axis=1, values=[z, domain_y])
    domain_y_conv = tf.reshape(domain_y, [FLAGS.batch_size, 1, 1, FLAGS.domain_num])

    h0 = tf.nn.relu(g_bn0(linear(z, 64*8*s16*s16, 'g_h0_lin'), train=False), name='g_h0_relu')
    h0 = tf.reshape(h0, [FLAGS.batch_size, s16, s16, 512])
    h0 = conv_cond_concat(h0, domain_y_conv)

    h1 = tf.nn.relu(g_bn1(deconv2d(h0,[FLAGS.batch_size, s8, s8, 256], name='g_h1'), train=False),name='g_h1_relu')
    h1 = conv_cond_concat(h1, domain_y_conv)

    h2 = tf.nn.relu(g_bn2(deconv2d(h1,[FLAGS.batch_size, s4, s4, 128], name='g_h2'), train=False),name='g_h2_relu')
    h2 = conv_cond_concat(h2, domain_y_conv)

    h3 = tf.nn.relu(g_bn3(deconv2d(h2,[FLAGS.batch_size, s2, s2, 64], name='g_h3'), train=False),name='g_h3_relu')
    h3 = conv_cond_concat(h3, domain_y_conv)

    h4 = tf.nn.tanh(deconv2d(h3, [FLAGS.batch_size, s, s, 3], name='g_h4'), name='g_h4_sigmoid')

    return h4
   

def inference(image, domain_y, random_z):

  G_image, reg_g  = generator(random_z, domain_y)

  D_logits_real, _ = discriminator(image, domain_y)

  D_logits_fake, reg_d = discriminator(G_image, domain_y, True)

  D_logits_fake_for_G, _ = discriminator(G_image, domain_y, True, True)

  return D_logits_real, D_logits_fake, D_logits_fake_for_G, reg_g, reg_d

def loss_l2(D_real, D_fake, D_fake_for_G, reg_g, reg_d, weight_G, weight_D):
  G_loss = tf.reduce_mean(tf.nn.l2_loss(D_fake_for_G - tf.zeros_like(D_fake_for_G))) 

  D_loss_real = tf.reduce_mean(tf.nn.l2_loss(D_real - tf.ones_like(D_real))) 

  D_loss_fake = tf.reduce_mean(tf.nn.l2_loss(D_fake + tf.ones_like(D_fake))) 

  D_loss = D_loss_real + D_loss_fake + reg_d * weight_D

  G_loss = G_loss + weight_G * reg_g

  return G_loss, D_loss


def loss(D_logits_real, D_logits_fake, D_logits_fake_for_G):

  G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
             logits=D_logits_fake_for_G, labels=tf.ones_like(D_logits_fake_for_G)))

  D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=D_logits_real, labels=tf.ones_like(D_logits_real)))

  D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=D_logits_fake, labels=tf.zeros_like(D_logits_fake)))

  D_loss = D_loss_real + D_loss_fake
  
  return G_loss, D_loss

def train(G_loss, D_loss, G_vars, D_vars, global_step):

  G_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) 
  D_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) 

  #G_train_op = G_optim.minimize(G_loss, var_list=G_vars)
  #D_train_op = D_optim.minimize(D_loss, var_list=D_vars)

  G_grads = G_optim.compute_gradients(G_loss, var_list=G_vars)
  D_grads = D_optim.compute_gradients(D_loss, var_list=D_vars)

  for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

  for grad, var in D_grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)
  for grad, var in G_grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  G_train_op = G_optim.apply_gradients(G_grads, global_step=global_step)
  D_train_op = D_optim.apply_gradients(D_grads)

  return G_train_op, D_train_op



