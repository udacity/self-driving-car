import tensorflow as tf


def weight_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    initial = initializer(shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


class ConvModel(object):
    ''' Implements the ConvNet model from the NVIDIA paper '''
    def __init__(self, dropout_prob=0.2, batch_norm=False, whitening=False, is_training=True):
        x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 1])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        x_image = x

        self.W_conv1 = weight_variable([5, 5, 3, 24])
        self.b_conv1 = bias_variable([24])
        self.h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1, 2) + self.b_conv1)
        if batch_norm:
            self.h_conv1 = tf.contrib.layers.batch_norm(self.h_conv1, is_training=is_training, trainable=True)

        self.W_conv2 = weight_variable([5, 5, 24, 36])
        self.b_conv2 = bias_variable([36])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_conv1, self.W_conv2, 2) + self.b_conv2)

        self.W_conv3 = weight_variable([5, 5, 36, 48])
        self.b_conv3 = bias_variable([48])
        self.h_conv3 = tf.nn.relu(conv2d(self.h_conv2, self.W_conv3, 2) + self.b_conv3)
        if batch_norm:
            self.h_conv3 = tf.contrib.layers.batch_norm(self.h_conv3, is_training=is_training, trainable=True)

        self.W_conv4 = weight_variable([3, 3, 48, 64])
        self.b_conv4 = bias_variable([64])
        self.h_conv4 = tf.nn.relu(conv2d(self.h_conv3, self.W_conv4, 1) + self.b_conv4)

        self.W_conv5 = weight_variable([3, 3, 64, 64])
        self.b_conv5 = bias_variable([64])
        self.h_conv5 = tf.nn.relu(conv2d(self.h_conv4, self.W_conv5, 1) + self.b_conv5)
        if batch_norm:
            self.h_conv5 = tf.contrib.layers.batch_norm(self.h_conv5, is_training=is_training, trainable=True)

        self.W_fc1 = weight_variable([1152, 1164])
        self.b_fc1 = bias_variable([1164])

        self.h_conv5_flat = tf.reshape(self.h_conv5, [-1, 1152])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv5_flat, self.W_fc1) + self.b_fc1)
        if batch_norm:
            self.h_fc1 = tf.contrib.layers.batch_norm(self.h_fc1, is_training=is_training, trainable=True)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob)

        self.W_fc2 = weight_variable([1164, 100])
        self.b_fc2 = bias_variable([100])
        self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2, name='fc2')
        if batch_norm:
            self.h_fc2 = tf.contrib.layers.batch_norm(self.h_fc2, is_training=is_training, trainable=True)
        self.h_fc2_drop = tf.nn.dropout(self.h_fc2, keep_prob)

        self.W_fc3 = weight_variable([100, 50])
        self.b_fc3 = bias_variable([50])
        self.h_fc3 = tf.nn.relu(tf.matmul(self.h_fc2_drop, self.W_fc3) + self.b_fc3, name='fc3')
        if batch_norm:
            self.h_fc3 = tf.contrib.layers.batch_norm(self.h_fc3, is_training=is_training, trainable=True)
        self.h_fc3_drop = tf.nn.dropout(self.h_fc3, keep_prob)

        self.W_fc4 = weight_variable([50, 10])
        self.b_fc4 = bias_variable([10])
        self.h_fc4 = tf.nn.relu(tf.matmul(self.h_fc3_drop, self.W_fc4) + self.b_fc4, name='fc4')
        if batch_norm:
            self.h_fc4 = tf.contrib.layers.batch_norm(self.h_fc4, is_training=is_training, trainable=True)
        self.h_fc4_drop = tf.nn.dropout(self.h_fc4, keep_prob)

        self.W_fc5 = weight_variable([10, 1])
        self.b_fc5 = bias_variable([1])
        y = tf.mul(tf.atan(tf.matmul(self.h_fc4_drop, self.W_fc5) + self.b_fc5), 2, name='y')

        self.x = x
        self.y_ = y_
        self.y = y
        self.keep_prob = keep_prob
        self.fc2 = self.h_fc2
        self.fc3 = self.h_fc3
