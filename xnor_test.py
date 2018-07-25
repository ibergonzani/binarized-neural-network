import tensorflow as tf
import numpy as np
import time

xnor_matmul_module = tf.load_op_library('./xnor_matmul/xnor_matmul.so')
print("xnor_matmul loaded")

xnor_gemm_module = tf.load_op_library('./xnor_matmul/xnor_gemm.so')
print("xnor_gemm loaded")

m = 2048
n = 2048
k = 2048

a_data = np.sign(np.random.rand(m, n))
b_data = np.sign(np.random.rand(n, k))


# operation definitions
a = tf.placeholder(tf.float32, (m, n))
b = tf.placeholder(tf.float32, (n, k))

stdn_mul = tf.matmul(a, b)
xnor_mul = xnor_matmul_module.xnor_matmul(a, b)
xnor_gem = xnor_gemm_module.xnor_gemm(a, b)

print(xnor_mul.get_shape())

trial = 5
exect = np.zeros((3, trial))

with tf.Session() as sess:
	for i in range(trial):
	
		start = time.time()
		stdn = sess.run(stdn_mul, feed_dict={a: a_data, b: b_data})
		exect[0, i] = time.time() - start
		
		start = time.time()
		xnor = sess.run(xnor_mul, feed_dict={a: a_data, b: b_data})
		exect[1, i] = time.time() - start
		
		start = time.time()
		gemm = sess.run(xnor_gem, feed_dict={a: a_data, b: b_data})
		exect[2, i] = time.time() - start
	
	diff_xnor = stdn - xnor
	diff_gemm = stdn - gemm
	print(np.max(diff_xnor), np.min(diff_xnor))
	print(np.max(diff_gemm), np.min(diff_gemm))

print('Execution times:\n', exect)
	
