import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
import commonDate as CD

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
layer1 = add_layer(xs, 64, 50, 'layer1', activation_function=tf.nn.tanh)
prediction = add_layer(layer1, 50, 10, 'layer2', activation_function=tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=[1]))  # loss
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)


with tf.Session() as sess:
    if tf.gfile.Exists(CD.FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(CD.FLAGS.summaries_dir)

    merged = tf.summary.merge_all()
    # summary writer goes in here
    train_writer = tf.summary.FileWriter(CD.FLAGS.summaries_dir+'train/', sess.graph)
    test_writer = tf.summary.FileWriter(CD.FLAGS.summaries_dir+'test/', sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(500):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.6})
        if i % 50 == 0:
            # record loss
            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)
