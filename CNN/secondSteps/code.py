import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


x = tf.placeholder( tf.float32, [None, 784])
y = tf.placeholder( tf.float32, [None, 10])

x_image  = tf.reshape(x , [-1, 28, 28, 1])

w_conv_1 = tf.Variable( tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv_1 = tf.Variable( tf.constant(0.1, shape=[32]))

conv_1   = tf.nn.conv2d(x_image, w_conv_1, strides=[1, 1, 1, 1], padding="SAME") + b_conv_1
h_conv_1 = tf.nn.relu(conv_1)

h_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

w_conv_2 = tf.Variable( tf.truncated_normal([5, 5, 32, 64], stddev=0.1)) 
b_conv_2 = tf.Variable( tf.constant(0.1, shape=[64]))


conv_2   = tf.nn.conv2d(h_pool_1, w_conv_2, strides=[1, 1, 1, 1], padding="SAME") + b_conv_2
h_conv_2 = tf.nn.relu(conv_2)

h_pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7*7*64])

w_fc_1   = tf.Variable( tf.truncated_normal([7*7*64, 1024], stddev=0.1) ) 
b_fc_1   = tf.Variable( tf.constant(0.1, shape=[1024]) ) 

h_fc_1   = tf.nn.relu( tf.matmul(h_pool_2_flat, w_fc_1) + b_fc_1)

keep_probability = tf.placeholder(tf.float32)
h_fc_dropout     = tf.nn.dropout(h_fc_1, keep_probability)

w_fc_2   = tf.Variable( tf.truncated_normal([1024, 10], stddev=0.1) ) 
b_fc_2   = tf.Variable( tf.constant(0.1, shape=[10]) )

logit_conv       = tf.matmul(h_fc_dropout, w_fc_2) + b_fc_2
y_conv   = tf.nn.softmax(logit_conv)

cross_entropy    = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logit_conv, labels=y))
train_step       = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

correct_predict  = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

init     = tf.global_variables_initializer()
sess     = tf.Session()

sess.run( init )

for i in range(10000):
	batch_xs, batch_ys = mnist.train.next_batch(64)
	sess.run(train_step, feed_dict={x : batch_xs, y : batch_ys, keep_probability : 0.5})
	print('Iteration' ,  i) 

batch_xs, batch_ys = mnist.test.next_batch(5000)
print(sess.run(accuracy,feed_dict ={x : batch_xs, y : batch_ys, keep_probability : 1.0 } ))