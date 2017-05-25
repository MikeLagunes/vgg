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

""" A VGG-16 implementation on TF-1.0
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import math

import tensorflow as tf

import vgg
import numpy as np
import os
import sys


import itertools
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', os.getcwd() + '/logs/vgg_train_rgb_16bs001lr_SGD_1p_data_1xranrot',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_integer('num_examples', 62504,
                            """Number of examples for evaluation to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

tf.app.flags.DEFINE_float('INITIAL_LEARNING_RATE', 0.01,
                          """Number of examples to run.""")

tf.app.flags.DEFINE_integer('batch_size', 24,
                            """Number of images to process in a batch.""")

IMAGE_SIZE = 227  # Taking full size

# Global constants describing the t-lessv2 data set.

NUM_CLASSES = 30
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1044  # 37584#18792#37584  # 4
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 62504#3455

EPOCHS_NUM = math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)

testing_dataset = ['/home/mikelf/Datasets/T-lessV2/shards/test_full/tless_test-00000-of-00004-full.tfrecords',
'/home/mikelf/Datasets/T-lessV2/shards/test_full/tless_test-00001-of-00004-full.tfrecords',
'/home/mikelf/Datasets/T-lessV2/shards/test_full/tless_test-00002-of-00004-full.tfrecords',
'/home/mikelf/Datasets/T-lessV2/shards/test_full/tless_test-00003-of-00004-full.tfrecords']

#testing_dataset =['/home/mikelf/Datasets/T-lessV2/shards/test_10percent/tless_test-00000-of-00004-10percent.tfrecords',
#'/home/mikelf/Datasets/T-lessV2/shards/test_10percent/tless_test-00001-of-00004-10percent.tfrecords',
#'/home/mikelf/Datasets/T-lessV2/shards/test_10percent/tless_test-00002-of-00004-10percent.tfrecords',
#'/home/mikelf/Datasets/T-lessV2/shards/test_10percent/tless_test-00003-of-00004-10percent.tfrecords']

EPOCHS_NUM = math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, cm[i, j],
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, index, shuffle):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch, index_batch = tf.train.shuffle_batch(
            [image, label, index],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch, index_batch = tf.train.batch(
            [image, label, index],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size]), tf.reshape(index_batch, [batch_size])


def read_and_decode(filename_queue, batch_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue, name='train')

    features = tf.parse_single_example(
        serialized_example,

        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/index': tf.FixedLenFeature([], tf.int64),
            'image/filename': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    image = tf.image.decode_image(features['image/encoded'])

    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    # *------------------- pre processing

    distorted_image = tf.cast(image, tf.float32)

    # Randomly flip the image horizontally.
    # distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # distorted_image = tf.image.random_brightness(distorted_image,
    #                                             max_delta=63)

    # distorted_image = tf.image.random_contrast(distorted_image,
    #                                           lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # *-------------------

    print("No shuffle Tensor Shape: ")
    print(image.get_shape())

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['image/class/label'], tf.int32)
    index = tf.cast(features['image/index'], tf.int32)

    im_filename = tf.cast(features['image/filename'], tf.string)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4

    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
    # print(filenm)



    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, label,
                                           min_queue_examples, batch_size, index,
                                           shuffle=False)


def read_and_decode_validation(filename_queue, batch_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue, name='validate')

    features = tf.parse_single_example(
        serialized_example,

        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/index': tf.FixedLenFeature([], tf.int64),
            'image/filename': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    image = tf.image.decode_image(features['image/encoded'])

    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    # *------------------- alternative preprocessing

    distorted_image = tf.cast(image, tf.float32)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    label = tf.cast(features['image/class/label'], tf.int32)
    index = tf.cast(features['image/index'], tf.int32)

    im_filename = tf.cast(features['image/filename'], tf.string)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                             min_fraction_of_examples_in_queue)

    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
    # print(filenm)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, label,
                                           min_queue_examples, batch_size, index,
                                           shuffle=False)


# Getting data for feeding------------------------------------------------------------------------------------------------------
filename_queue_testing = tf.train.string_input_producer(testing_dataset, name='queue_runner3', shuffle=False)

images_p, labels_p, indexs_p = read_and_decode_validation(filename_queue_testing, batch_size=FLAGS.batch_size)


# ----------------------------------------------------------------------------------------------------------------------------

def train():
    sys.stdout.write("\033[93m")  # yellow message

    print("Load and test model")

    sys.stdout.write("\033[0;0m")

    with tf.Session() as sess:

        global_step = tf.contrib.framework.get_or_create_global_step()

        images = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
        labels = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
        indexes = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
        # mode_eval = tf.placeholder(tf.bool, shape=())
        keep_prob = tf.placeholder(tf.float32)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = vgg.inference(images, keep_prob)

        # Calculate loss.
        loss = vgg.loss(logits, labels)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        prediction = tf.argmax(logits, 1)

        #cmatix = tf.contrib.metrics.confusion_matrix(prediction, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = vgg.train(loss, global_step)

        # train = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Restore the moving average version of the learned variables for eval.

        #variable_averages = tf.train.ExponentialMovingAverage(
        #    vgg.MOVING_AVERAGE_DECAY)
        #variables_to_restore = variable_averages.variables_to_restore()
        #saver = tf.train.Saver(variables_to_restore)


        #saver = tf.train.import_meta_graph('/home/mikelf/Datasets/T-lessV2/restore_models/model.ckpt-61000.meta')


        saver.restore(sess, "/home/mikelf/Datasets/T-lessV2/restore_models/vgg_train_rgb_16bs001lr_SGD_50p_data_x1randrot_2nd/model.ckpt-78000")

        #sess.run(tf.global_variables_initializer())


        print("Model restored.")

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()


        # Build an initialization operation to run below.
        ##init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        # sess = tf.Session(config=tf.ConfigProto(
        #    log_device_placement=FLAGS.log_device_placement))

        ##sess.run(init)

        coord = tf.train.Coordinator()

        # Start the queue runners.
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        summary_writer_train = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        # summary_writer_validation = tf.summary.FileWriter(FLAGS.validate_dir)



        loss_train = np.array([])
        loss_valid = np.array([])
        precision_test = np.array([])

        steps_train = np.array([])
        steps_valid = np.array([])
        steps_precision = np.array([])

        EPOCH = 0
        start_time_global = time.time()

        print("getting precision on test dataset")

        # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        # feeding data for evaluation

        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        x = []
        cf_matrix_array = []
        labels_array = []

        while step < num_iter:
            images_batch, labels_batch, index_batch = sess.run([images_p, labels_p, indexs_p])

            predictions, cf_matrix = sess.run([top_k_op, prediction],
                                   feed_dict={images: images_batch, labels: labels_batch, indexes: index_batch,
                                              keep_prob: 1.0})

            true_count += np.sum(predictions)
            step += 1
            x.extend(index_batch)
            cf_matrix_array = np.append(cf_matrix_array, cf_matrix, axis=0)
            labels_array = np.append(labels_array, labels_batch, axis=0)




        print(cf_matrix_array.shape)

        print(len(x))
        dupes = [xa for n, xa in enumerate(x) if xa in x[:n]]
        # print(sorted(dupes))
        print(len(dupes))

        precision = true_count / total_sample_count

        print('%s: precision @ 1 = %.5f' % (datetime.now(), precision))

        precision_test = np.concatenate((precision_test, [precision]))
        steps_precision = np.concatenate((steps_precision, [EPOCH]))


        final_time_global = time.time()

        print("Finish")

        print(final_time_global - start_time_global)

        cnf_matrix = confusion_matrix(labels_array, cf_matrix_array)
        np.set_printoptions(precision=2)

        class_names = ['1',  '2',   '3',  '4',  '5',  '6',  '7',  '8',  '9', '10',
                       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                       '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        #plt.figure()
        #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
        #                      title='Normalized confusion matrix')

        plt.show()

        coord.request_stop()
        coord.join(threads)

        sess.close()



        # ================================


def main(argv=None):
    # vgg.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()