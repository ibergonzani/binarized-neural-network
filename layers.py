import tensorflow as tf
from tensorflow.python.framework import ops



def binarize(x):
	# we also have to reassign the sign gradient otherwise it will be almost everywhere equal to zero
	# using the straight through estimator
	with tf.get_default_graph().gradient_override_map({'Sign': 'Identity'}):
		#return tf.sign(x)				#	<-- wrong sign doesn't return +1 for zero
		return tf.sign(tf.sign(x)+1e-8) #	<-- this should be ok, ugly but okay


	
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



def binaryConv2d(inputs, filters, kernel_size, strides, padding="VALID", use_bias=True, activation=None, binarize_input=True, trainable=True, 
					reuse=False, use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1,1,1,1], name='binaryconv2d'):
	
	assert len(strides) == 2
	assert data_format in ['NHWC', 'NCHW']
	
	if data_format == 'NHWC':
		strides = [1] + strides + [1]
		in_ch = inputs.get_shape().as_list()[3]
		wshape = kernel_size + [in_ch, filters]
	elif data_format == 'NCHW':
		strides = [1, 1] + strides
		in_ch = inputs.get_shape().as_list()[1]	
		wshape = [in_ch, filters] + kernel_size
	
	
	with tf.variable_scope(name, reuse=reuse):
		# getting filter weights and add clip operation on them (between -1, 1)			
		fw = tf.get_variable('weight', wshape, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
		fw = tf.clip_by_value(fw, -1, 1)
		
		# binarize  input and weights of the layer
		if binarize_input:
			inputs = binarize(inputs)
		fw = binarize(fw)
		
		# adding convolution
		out = tf.nn.conv2d(inputs, fw, strides, padding, use_cudnn_on_gpu=use_cudnn_on_gpu, 
							data_format=data_format, dilations=dilations)
		
		# adding bias
		if use_bias:
			fb = tf.get_variable('bias', [filters], initializer=tf.zeros_initializer, trainable=trainable)
			out = tf.nn.bias_add(out, fb, data_format=data_format)
		
		# applying activation function
		if activation:
			out = activation(out, 'activation')
		
		# activation collection is not automatically populated
		tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, out)
		
		return out


	
# compute the approximate power of 2 of the input x
# via hardware it is as simple as get the index of the most significative bit
def ap2(x):
	return tf.sign(x) * tf.pow(2.0, tf.round(tf.log(tf.abs(x)) / tf.log(2.0)))

	

	

##################################################################################################
############################################ NOTE ################################################
##################################################################################################
## The functions below computing batch normalizations should use shifting operations instead of ##
## multiplications and divisions. Clearly the shift operations are only implemented for integer ##
## variables. For this reason the shift based batch normalizations are still implemented with   ##
## multiplications and divisions but using only the correspondent approximated power of two in  ##
## place of the op. right var (obtaining the same result of a shifting).						##
## Consequently these shift based batch normalizations have a smaller precision than standars	##
## methods and are slower than the hypotetical implementations with hardware accelerated float  ##
## shifting. They are used only to show their behaviour.										##
##################################################################################################
##################################################################################################
 
# Shift based Batch Normalizing Transform, applied to activation (x) over a mini-batch,
#as described in http://arxiv.org/abs/1502.03167
def shift_batch_norm(x, training=True, momentum=0.99, epsilon=1e-8, reuse=False, name="batch_norm"):
	
	xshape = x.get_shape()[1:]
	
	with tf.variable_scope(name, reuse=reuse):
		gamma = tf.get_variable('gamma', xshape, initializer=tf.ones_initializer, trainable=True)
		beta  = tf.get_variable('beta', xshape, initializer=tf.zeros_initializer, trainable=True)
		
		mov_avg = tf.get_variable('mov_avg', xshape, initializer=tf.zeros_initializer, trainable=False)
		mov_var = tf.get_variable('mov_std', xshape, initializer=tf.ones_initializer, trainable=False)
		
		def training_xdot():
			avg = tf.reduce_mean(x, axis=0)							# feature means
			cx = x - avg											# centered input
			var = tf.reduce_mean(tf.multiply(cx, ap2(cx)), axis=0)	# apx variance
			
			# updating ops. for moving average and moving variance used at inference time
			avg_update = tf.assign(mov_avg, momentum * mov_avg + (1.0 - momentum) * avg)
			var_update = tf.assign(mov_var, momentum * mov_var + (1.0 - momentum) * var)
			
			with tf.control_dependencies([avg_update, var_update]):
				return cx / ap2(tf.sqrt(var + epsilon))				# normalized input
			
		def inference_xdot():
			return (x - mov_avg) / ap2(tf.sqrt(mov_var + epsilon))
		
		xdot = tf.cond(training, training_xdot, inference_xdot)
		out = tf.multiply(ap2(gamma), xdot) + beta					# scale and shift input distribution
	
	return out


# Spatial shift based batch normalization, like spatial batch normalization it keeps
# the convolution property. Hence it applies the same transformation to each element
# of the same feature map
def spatial_shift_batch_norm(x, data_format='NHWC', training=True, momentum=0.99, epsilon=1e-8, reuse=False, name="spatial_batch_norm"):
	assert data_format in ['NHWC', 'NCHW']
	
	if data_format == "NHWC":
		mean_axis = (0,1,2)
		channel_axis = 3
		ch_tensor_shape = [1, 1, 1, x.get_shape().as_list()[channel_axis]]
	elif data_format == "NCHW":
		mean_axis = (0,2,3)
		channel_axis = 1
		ch_tensor_shape = [1, x.get_shape().as_list()[channel_axis], 1, 1]
	
	with tf.variable_scope(name, reuse=reuse):
		gamma = tf.get_variable('gamma', ch_tensor_shape, initializer=tf.ones_initializer, trainable=True)
		beta  = tf.get_variable('beta', ch_tensor_shape, initializer=tf.zeros_initializer, trainable=True)
		
		mov_avg = tf.get_variable('mov_avg', ch_tensor_shape, initializer=tf.zeros_initializer, trainable=False)
		mov_var = tf.get_variable('mov_std', ch_tensor_shape, initializer=tf.ones_initializer, trainable=False)
		
		def training_xdot():
			avg = tf.reduce_mean(x, axis=mean_axis, keepdims=True)
			cx = x - avg																	# centered input
			var = tf.reduce_mean(tf.multiply(cx, ap2(cx)), axis=mean_axis, keepdims=True)	# apx variance
			
			# updating ops. for moving average and moving variance used at inference time
			avg_update = tf.assign(mov_avg, momentum * mov_avg + (1.0 - momentum) * avg)
			var_update = tf.assign(mov_var, momentum * mov_var + (1.0 - momentum) * var)
			
			with tf.control_dependencies([avg_update, var_update]):
				return cx / ap2(tf.sqrt(var + epsilon))					# normalized input
		
		def inference_xdot():
			return (x - mov_avg) / ap2(tf.sqrt(mov_var + epsilon))
		
		xdot = tf.cond(training, training_xdot, inference_xdot)
		out = tf.multiply(ap2(gamma), xdot) + beta					# scale and shift input distribution
	
	return out
	