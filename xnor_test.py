import tensorflow as tf
import numpy as np
import argparse
import time
import layers
from tensorflow.python.client import timeline


parser = argparse.ArgumentParser(description='')
parser.add_argument('--tracing', default=False, action='store_true', help='Save multiplication execution tracing')
args = parser.parse_args()



xnor_matmul_module = tf.load_op_library('./xnor_matmul/xnor_matmul.so')
print("xnor_matmul loaded")



def random_xnor_network(ishape, oshape, wi, wh, wo, hsize=1024, hlayers=3, group64=False):
	
	x = tf.placeholder(tf.float32, ishape)

	y = tf.clip_by_value(tf.matmul(x, layers.binarize(wi)), -1.0, 1.0)
	
	out = y
	for l in range(hlayers):
		y = xnor_matmul_module.xnor_matmul(y, layers.binarize(wh), group64=group64)
		y = layers.binarize(tf.clip_by_value(y, -1.0, 1.0))
		
	y = tf.matmul(y, layers.binarize(wo))
	
	return x, y
	
	
def random_mult_network(ishape, oshape, wi, wh, wo, hsize=1024, hlayers=3):
	
	x = tf.placeholder(tf.float32, ishape)
	
	y = tf.clip_by_value(tf.matmul(x, layers.binarize(wi)), -1.0, 1.0)
	y = layers.binarize(y)
	
	for l in range(hlayers):
		y = tf.matmul(y, layers.binarize(wh))
		y = layers.binarize(tf.clip_by_value(y, -1.0, 1.0))
		
	y = tf.matmul(y, layers.binarize(wo))
	
	return x, y



batch_size = 1024
input_shape = [None, 1024]
output_shape = [None, 10]
hidden_units = 4096
hidden_layers = 3

# testing networks with same weights
wi_data = np.sign(np.random.rand(input_shape[1], hidden_units) - 0.5)
wh_data = np.sign(np.random.rand(hidden_units, hidden_units) - 0.5)
wo_data = np.sign(np.random.rand(hidden_units, output_shape[1]) - 0.5)

wi = tf.Variable(wi_data, dtype=tf.float32)
wh = tf.Variable(wh_data, dtype=tf.float32)
wo = tf.Variable(wo_data, dtype=tf.float32)

xnor32_net_input, xnor32_net_output = random_xnor_network(input_shape, output_shape, wi, wh, wo, hidden_units, hidden_layers, group64=False)
xnor64_net_input, xnor64_net_output = random_xnor_network(input_shape, output_shape, wi, wh, wo, hidden_units, hidden_layers, group64=True)
stdmul_net_input, stdmul_net_output = random_mult_network(input_shape, output_shape, wi, wh, wo, hidden_units, hidden_layers)

number_of_tests = 5
exect = np.zeros((3, number_of_tests))

# simulation of running through a dataset of size equal to mnist's
with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())
	
	for test in range(number_of_tests):
		print("test iteration", test)
		
		X = np.sign(np.random.rand(batch_size, input_shape[1]))
		
		print("xnor 32")
		start = time.time()
		for i in range(int(60000/batch_size)):
			xnor32 = sess.run(xnor32_net_output, feed_dict={xnor32_net_input: X})
		exect[0, test] = time.time() - start
		
		print("xnor 64")
		start = time.time()
		for i in range(int(60000/batch_size)):
			xnor64 = sess.run(xnor64_net_output, feed_dict={xnor64_net_input: X})
		exect[1, test] = time.time() - start
		
		print("std mul")
		start = time.time()
		for i in range(int(60000/batch_size)):
			stdn = sess.run(stdmul_net_output, feed_dict={stdmul_net_input: X})
		exect[2, test] = time.time() - start
		
	# print(xnor32.shape, xnor64.shape, stdn.shape)
	# print(np.sum(xnor32 == 0), np.sum(xnor64 == 0), np.sum(stdn == 0))
	print("Result difference stdn-xnor32:", np.sum(np.abs(stdn - xnor32)))
	print("Result difference stdn-xnor64:", np.sum(np.abs(stdn - xnor64)))

print("\nMLP network execution times for xnor32, xnor64, stdmul:")
print(exect, "\n\n")


print("TESTING SINGLE MULTIPLICATION:\n")
		
sizes = [1024, 2048, 4096, 8192]
trial = 5
exect = {}
mop = {}


for size in sizes:

	m = size
	n = size
	k = size
	
	# operation definitions
	a = tf.placeholder(tf.float32, (m, n))
	b = tf.placeholder(tf.float32, (n, k))

	stdn_mul = tf.matmul(a, b)
	xnor_mul32 = xnor_matmul_module.xnor_matmul(a, b, group64=False)
	xnor_mul64 = xnor_matmul_module.xnor_matmul(a, b, group64=True)
	
	mop[size] = {'std':stdn_mul, 'x32':xnor_mul32, 'x64':xnor_mul64, 'a':a, 'b':b}
	
	exect[size] = np.zeros((3, trial))

	

with tf.Session() as sess:
	
	summary_writer = tf.summary.FileWriter('xnor_matmul/logs/', sess.graph)
	sess.run(tf.global_variables_initializer())
	
	for size in sizes:
		
		a_data = np.sign(np.random.rand(size, size)-0.5)
		b_data = np.sign(np.random.rand(size, size)-0.5)
		
		for i in range(trial):
		
			start = time.time()
			stdn = sess.run(mop[size]['std'], feed_dict={mop[size]['a']: a_data, mop[size]['b']: b_data})
			exect[size][0, i] = time.time() - start
			
			start = time.time()
			xnor32 = sess.run(mop[size]['x32'], feed_dict={mop[size]['a']: a_data, mop[size]['b']: b_data})
			exect[size][1, i] = time.time() - start
			
			start = time.time()
			xnor64 = sess.run(mop[size]['x64'], feed_dict={mop[size]['a']: a_data, mop[size]['b']: b_data})
			exect[size][2, i] = time.time() - start
			
			# tracing goes in conflict with nvprof (just don't use them together)
			if args.tracing:
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				
				_, _, _ = sess.run([mop[size]['std'], mop[size]['x32'], mop[size]['x64']], 
									feed_dict={mop[size]['a']: a_data, mop[size]['b']: b_data}, options=run_options, run_metadata=run_metadata)
				summary_writer.add_run_metadata(run_metadata, '%s size_%d step_%d' % ('stdn-xnor32-xnor64', size, i))
				
				fetched_timeline = timeline.Timeline(run_metadata.step_stats)
				chrome_trace = fetched_timeline.generate_chrome_trace_format()
				with open('xnor_matmul/logs/tracing/timeline%dx%d_run%d.json' % (size,size,i), 'w') as f:
					f.write(chrome_trace)

		# results check
		# print(stdn.shape, xnor32.shape, xnor64.shape)
		diff_xnor32 = stdn - xnor32
		diff_xnor64 = stdn - xnor64
		print(size, 'Difference xnor 32:', np.max(diff_xnor32), np.min(diff_xnor32))
		print(size, 'Difference xnor 64:', np.max(diff_xnor64), np.min(diff_xnor64))
	
	
	summary_writer.close()

for size in sizes:
	print('\nExecution times std-xnor32-xnor64 (%dx%d):\n' % (size, size), exect[size])
	print('Mean times:', np.mean(exect[size][:,1:], axis=1))

