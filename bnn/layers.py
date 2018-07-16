import tensorflow as tf


def binarize(x):
	# we also have to reassign the sign gradient otherwise it will be almost everywhere equal to zero
	# using the straight through estimator
	with tf.override_gradient_map('Sign', 'Identity'):
		return tf.sign(x)


	
def binaryDense(x, trainable, binarize_input=True, activation=None):
	
	# getting layer weights and add clip operation (between -1, 1)
	w = tf.get_variable('weight', initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
	w = tf.clip_by_value(w, -1, 1)

	# binarize input and weights of the layer
	if binarize_input:
		x = binarize(x)
	w = binarize(w)
	
	# adding layer operations -> (w*x + b)
	out = tf.matmul(x, w)
	if bias:
		b = tf.get_variable('bias', initializer=, trainable=trainable)
		out = out + b
	
	# applying activation function
	if activation:
		out = activation(out)
	
	# activations collection is not automatically populated
	tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, out)
	return out



def binaryConv2d(input, filters, kernel_size, strides, padding, binarize_input=True, trainable=True, reuse=False,
					activation=None, use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1,1,1,1], name='binaryconv2d'):
	
	with tf.variable_scope(name, reuse=reuse):
		# getting filter weights and add clip operation on them (between -1, 1)			
		fw = tf.get_variable('weight', [filters] + [kernel_size, 3], 
								initializer=tf.contrib.layers.conv2d_initializer(), trainable=trainable)
		fw = tf.clip_by_value(w, -1, 1)
		
		# binarize  input and weights of the layer
		if binarize_input:
			input = binarize(input)
		fw = binarize(fw)
		
		# adding convolution
		out = tf.nn.conv2d(input, fw, strides, padding, use_cudnn_on_gpu=use_cudnn_on_gpu, 
							data_format=data_format, dilations=dilations)
		
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