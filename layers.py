import tensorflow as tf
from tensorflow.python.framework import ops

def binarize(x):
	# we also have to reassign the sign gradient otherwise it will be almost everywhere equal to zero
	# using the straight through estimator
	with tf.get_default_graph().gradient_override_map({'Sign': 'Identity'}):
		return tf.sign(x)


	
def binaryDense(inputs, units, activation=None, use_bias=True, trainable=True, binarize_input=True, name='binarydense', reuse=False):
	
	# flatten the input 
	flat_input = tf.contrib.layers.flatten(inputs)
	
	# count all the input units (thus check the shape without considering the batch size)
	in_units = flat_input.get_shape().as_list()[1]
	
	with tf.variable_scope(name, reuse=reuse):
		# getting layer weights and add clip operation (between -1, 1)
		w = tf.get_variable('weight', [in_units, units], initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
		w = tf.clip_by_value(w, -1, 1)

		# binarize input and weights of the layer
		if binarize_input:
			flat_input = binarize(flat_input)
		w = binarize(w)
		
		# adding layer operation -> (w*x + b)
		out = tf.matmul(flat_input, w)
		if use_bias:
			b = tf.get_variable('bias', [units], initializer=tf.zeros_initializer(), trainable=trainable)
			out = tf.nn.bias_add(out, b)
		
		# applying activation function
		if activation:
			out = activation(out)
		
		# activations collection is not automatically populated
		tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, out)
		return out



def binaryConv2d(inputs, filters, kernel_size, strides, padding="VALID", bias=True, activation=None, binarize_input=True, trainable=True, 
					reuse=False, use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1,1,1,1], name='binaryconv2d'):
	
	assert len(strides) == 2
	assert data_format in ['NHWC', 'NCHW']
	
	if data_format == 'NHWC':
		strides = [1] + strides + [1]
		in_ch = input.get_shape().as_list()[3]
		wshape = [in_ch] + kernel_size + [filters]
	elif data_format == 'NCHW':
		strides = [1, 1] + strides
		in_ch = input.get_shape().as_list()[1]	
		wshape = [in_ch, filters] + kernel_size
	
	
	with tf.variable_scope(name, reuse=reuse):
		# getting filter weights and add clip operation on them (between -1, 1)			
		fw = tf.get_variable('weight', wshape, trainable=trainable, initializer=tf.contrib.layers.conv2d_initializer())
		fw = tf.clip_by_value(w, -1, 1)
		
		# binarize  input and weights of the layer
		if binarize_input:
			inputs = binarize(inputs)
		fw = binarize(fw)
		
		# adding convolution
		out = tf.nn.conv2d(inputs, fw, strides, padding, use_cudnn_on_gpu=use_cudnn_on_gpu, 
							data_format=data_format, dilations=dilations)
		
		# adding bias
		if bias:
			fb = tf.get_variable('bias', [filters], initializer=tf.zeros_initializer, trainable=trainable)
			out = tf.nn.bias_add(out, fb, data_format=data_format)
		
		# applying activation function
		if activation:
			out = activation(out, 'activation')
		
		# activation collection is not automatically populated
		tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, out)
		return out

		



# redefine gradient application

# gradient = optimizer.compute_gradient()
# binarized_gradient = 




# batch normalization can be added using tf.contrib.batch_norm
# as described in http://arxiv.org/abs/1502.03167
# or by using the following function

# # Shift based Batch Normalizing Transform, applied to activation (x) over a mini-batch.
# def batchnormalization(x):
	# # centered input
	# cx = c - tf.mean(x, axis=1)
	# #apx variance
	# apx_var = tf.variance(x)
	
	# #tf.bitwise.right_shift 
	
	# out = 
	
	# return out