import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
from tensorflow.commonDate import *

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name+'/Weights', Weights)
        with tf.name_scope('biase'):
            biase = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name + '/biase', biase)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biase
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracyResult = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    tf.summary.histogram('accuracyResult', accuracyResult)
    return accuracyResult

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, n_layer=1, activation_function=tf.nn.softmax)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),axis=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.summaries_dir, sess.graph)
    sess.run(init)

    for step in range(1001):
        batch_xs, batch_ys = mnist.train.next_batch(200)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if step%50 == 0:
            result = sess.run(merged, feed_dict={xs:batch_xs, ys:batch_ys})
            writer.add_summary(result, step)
            print(compute_accuracy(mnist.test.images, mnist.test.labels))






