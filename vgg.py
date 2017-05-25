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

"""Builds the CIFAR-10 network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
# Basic model parameters.

NUM_CLASSES = 30
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1044#18792#37584  # 4
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3455

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 30  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay_conv(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        initializer=tf.contrib.layers.xavier_initializer_conv2d())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _variable_with_weight_decay_fc(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        initializer=tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images, keep_prob):
    """Build the CIFAR-10 model.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #

    # conv1   ===================================================================

    with tf.variable_scope('conv1_1') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv1_1',
                                             shape=[3, 3, 3, 64],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv1_1', [64], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1_1)

    with tf.variable_scope('conv1_2') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv1_2',
                                             shape=[3, 3, 64, 64],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv1_2', [64], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1_2)

    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv2   ====================================================================

    with tf.variable_scope('conv2_1') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv2_1',
                                             shape=[3, 3, 64, 128],
                                             stddev=1e-1,
                                             wd=None)

        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv2_1', [128], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2_1)

    with tf.variable_scope('conv2_2') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv2_2',
                                             shape=[3, 3, 128, 128],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv2_2', [128], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2_2)

    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # conv3   ====================================================================

    with tf.variable_scope('conv3_1') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv3_1',
                                             shape=[3, 3, 128, 256],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv3_1', [256], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(pre_activation, name=scope.name)

        _activation_summary(conv3_1)

    with tf.variable_scope('conv3_2') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv3_2',
                                             shape=[3, 3, 256, 256],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv3_2', [256], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv3_2)

    with tf.variable_scope('conv3_3') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv3_3',
                                             shape=[3, 3, 256, 256],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv3_3', [256], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv3_3)

    # pool3
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # conv4   ====================================================================

    with tf.variable_scope('conv4_1') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv4_1',
                                             shape=[3, 3, 256, 512],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv4_1', [512], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv4_1)

    with tf.variable_scope('conv4_2') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv4_2',
                                             shape=[3, 3, 512, 512],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv4_2', [512], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv4_2)

    with tf.variable_scope('conv4_3') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv4_3',
                                             shape=[3, 3, 512, 512],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv4_3', [512], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv4_3)

    # pool4
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    # conv5   ====================================================================

    with tf.variable_scope('conv5_1') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv5_1',
                                             shape=[3, 3, 512, 512],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv5_1', [512], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv5_1)

    with tf.variable_scope('conv5_2') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv5_2',
                                             shape=[3, 3, 512, 512],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv5_2', [512], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv5_2)

    with tf.variable_scope('conv5_3') as scope:
        kernel = _variable_with_weight_decay_conv('weights_conv5_3',
                                             shape=[3, 3, 512, 512],
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases_conv5_3', [512], tf.constant_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv5_3)

    # pool5
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    # FC1  ====================================================================

    with tf.variable_scope('fc1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay_fc('weights_fc1', shape=[dim, 4096],
                                              stddev=1e-1, wd=None)
        biases = _variable_on_cpu('biases_fc1', [4096], tf.constant_initializer())

        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        fc1 = tf.nn.dropout(fc1, keep_prob)

        _activation_summary(fc1)

    # FC2   ====================================================================


    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay_fc('weights_fc2', shape=[4096, 4096],
                                              stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases_fc2', [4096], tf.constant_initializer())

        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

        fc2 = tf.nn.dropout(fc2, keep_prob)

        _activation_summary(fc2)

    # FC3   ====================================================================


    with tf.variable_scope('fc3') as scope:
        weights = _variable_with_weight_decay_fc('weights_fc3', [4096, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=None)
        biases = _variable_on_cpu('biases_fc3', [NUM_CLASSES],
                                  tf.constant_initializer())

        fc3 = tf.matmul(fc2, weights) + biases

        _activation_summary(fc3)

    return fc3


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    #labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)


    #if counter < 5: INITIAL_LEARNING_RATE = 0.0000001

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):

        opt = tf.train.GradientDescentOptimizer(lr)
        #opt = tf.train.RMSPropOptimizer(lr)
        #opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
