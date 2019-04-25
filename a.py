import tensorflow as tf
import numpy as np

a = tf.constant([5, 5, 5, 5])
print(a)
print(tf.split(a, 2))

b = tf.placeholder(dtype=tf.float32, shape=[None, 5])
n = tf.shape(b)[0]
c = b[0:0+n:2]

sess = tf.Session()
b_val = np.random.rand(4, 5)
print(b_val)
c = sess.run(c, feed_dict={b: b_val})
print(c)
