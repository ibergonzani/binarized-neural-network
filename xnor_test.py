import tensorflow as tf
import numpy as np

xnor_matmul_module = tf.load_op_library('./xnor_matmul/xnor_matmul.so')
print("Operation loaded")

m = 256
n = 512
k = 512

a_data = np.sign(np.random.rand(m, n))
b_data = np.sign(np.random.rand(n, k))


# operation definitions
a = tf.placeholder(tf.float32, (m, n))
b = tf.placeholder(tf.float32, (n, k))

stdn_mul = tf.matmul(a, b)
xnor_mul = xnor_matmul_module.xnor_matmul(a, b)

print(xnor_mul.get_shape())

with tf.Session() as sess:
	stdn = sess.run(stdn_mul, feed_dict={a: a_data, b: b_data})
	xnor = sess.run(xnor_mul, feed_dict={a: a_data, b: b_data})
	
	diff = stdn - xnor
	print(np.max(diff), np.min(diff))
	
