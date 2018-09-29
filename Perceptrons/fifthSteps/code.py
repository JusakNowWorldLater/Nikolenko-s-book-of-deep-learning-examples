import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


def fullytconnected_layer(tensor, input_size, out_size):
	W = tf.Variable( tf.truncated_normal([input_size, out_size], stddev = 0.1) )
	B = tf.Variable( tf.truncated_normal([out_size], stddev=0.1))
	return tf.nn.tanh( tf.matmul( tensor , W ) + B )


def batchhorm_layer(tensor, size):
	batch_mean, batch_var = tf.nn.moments(tensor,[0])
	beta = tf.Variable( tf.zeros([size]) )
	scale = tf.Variable( tf.ones([size]) )
	return tf.nn.batch_normalization( tensor, batch_mean, batch_var, beta, scale, 0.001 )


y = tf.placeholder( tf.float32, [None, 10] )

x = tf.placeholder( tf.float32, [None, 784] )
h1 = fullytconnected_layer(x, 784, 100)
h1_b = batchhorm_layer( h1, 100)
h2 = fullytconnected_layer(h1_b, 100, 100)
y_logit = fullytconnected_layer(h2, 100, 10)

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

correct_prediction = tf.equal( tf.argmax(y,1), tf.argmax(y_logit,1) )
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run( init )

for i in range(100) :
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_op, feed_dict={ x : batch_xs, y : batch_ys }) 

	print("Accuracy: %s" %
		sess.run(accuracy, feed_dict={ x : mnist.test.images, y : mnist.test.labels}) )