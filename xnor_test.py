import tensorflow as tf
import numpy as np
import time
from tensorflow.python.client import timeline

xnor_matmul_module = tf.load_op_library('./xnor_matmul/xnor_matmul.so')
print("xnor_matmul loaded")

xnor_gemm_module = tf.load_op_library('./xnor_matmul/xnor_gemm.so')
print("xnor_gemm loaded")

m = 1024
n = 2048
k = 1024

a_data = np.sign(np.random.rand(m, n)-0.5)
b_data = np.sign(np.random.rand(n, k)-0.5)

# print(a_data, b_data)

# operation definitions
a = tf.placeholder(tf.float32, (m, n))
b = tf.placeholder(tf.float32, (n, k))

stdn_mul = tf.matmul(a, b)
xnor_mul = xnor_matmul_module.xnor_matmul(a, b, group64=True)
# xnor_gem = xnor_gemm_module.xnor_gemm(a, b)

print(xnor_mul.get_shape())

trial = 5
exect = np.zeros((3, trial))

with tf.Session() as sess:
	
	summary_writer = tf.summary.FileWriter('xnor_matmul/logs/', sess.graph)
	
	sess.run(tf.global_variables_initializer())
	
	for i in range(trial):
	
		start = time.time()
		stdn = sess.run(stdn_mul, feed_dict={a: a_data, b: b_data})
		exect[0, i] = time.time() - start
		
		start = time.time()
		xnor = sess.run(xnor_mul, feed_dict={a: a_data, b: b_data})
		exect[1, i] = time.time() - start
		
		# start = time.time()
		# gemm = sess.run(xnor_gem, feed_dict={a: a_data, b: b_data})
		# exect[2, i] = time.time() - start
		
		
		# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		# run_metadata = tf.RunMetadata()
		
		# _, _ = sess.run([stdn_mul, xnor_mul], feed_dict={a: a_data, b: b_data}, options=run_options, run_metadata=run_metadata)
		# summary_writer.add_run_metadata(run_metadata, '%s step%d' % ('stdn-xnor', i))
		
		# fetched_timeline = timeline.Timeline(run_metadata.step_stats)
		# chrome_trace = fetched_timeline.generate_chrome_trace_format()
		# with open('xnor_matmul/logs/tracing/timeline_s%d.json' % i, 'w') as f:
			# f.write(chrome_trace)

	
	print(stdn.shape)
	print(xnor.shape)
	# print(gemm.shape)
	diff_xnor = stdn - xnor
	# diff_gemm = stdn - gemm
	print('Difference xnor:', np.max(diff_xnor), np.min(diff_xnor))
	# print('Difference gemm:', np.max(diff_gemm), np.min(diff_gemm))
	
	summary_writer.close()

print('Execution times:\n', exect)

