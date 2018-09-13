import tensorflow as tf
import layers

		
def multilayer_perceptron(input, units_list):
	output = input
	for l in range(len(units_list)):
		output = tf.layers.dense(output, units_list[l], activation=tf.nn.tanh)
		
	return input, output
	
	
def binary_multilayer_perceptron(input, units_list):
	output = input
	for l in range(len(units_list)-1):
		output = layers.binaryDense(output, units_list[l], activation=None, name='binarydense'+str(l))
		output = tf.contrib.layers.batch_norm(output)
		output = tf.clip_by_value(output, -1, 1)
	output = layers.binaryDense(output, units_list[l+1], activation=None, name='binarydense'+str(len(units_list)-1))
	output = tf.contrib.layers.batch_norm(output)
	return input, output

	
def cifar10(input, training=True):
	out = tf.layers.conv2d(input, 128, [3,3], [1,1], padding='VALID', use_bias=False, name='c_conv2d_1')
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.conv2d(out, 128, [3,3], [1,1], padding='SAME', use_bias=False, name='conv2d_1')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.conv2d(out, 256, [3,3], [1,1], padding='SAME', use_bias=False, name='conv2d_2')
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.conv2d(out, 256, [3,3], [1,1], padding='SAME', use_bias=False, name='conv2d_3')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.conv2d(out, 512, [3,3], [1,1], padding='SAME', use_bias=False, name='conv2d_4')
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.conv2d(out, 512, [3,3], [1,1], padding='SAME', use_bias=False, name='conv2d_5')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.flatten(out)
	out = tf.layers.dense(out, 1024, use_bias=False, name='dense_1')
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.dense(out, 1024, use_bias=False, name='dense_2')
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	output = tf.layers.dense(out, 10, name='dense_3')
	
	return input, output

	
def binary_cifar10(input):
	out = layers.binaryConv2d(input, 128, [3,3], [1,1], padding='VALID', use_bias=False, binarize_input=False, name='bc_conv2d_1')
	out = tf.contrib.layers.batch_norm(out)
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 128, [3,3], [1,1], padding='SAME', use_bias=False, name='bnn_conv2d_1')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.contrib.layers.batch_norm(out)
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 256, [3,3], [1,1], padding='SAME', use_bias=False, name='bnn_conv2d_2')
	out = tf.contrib.layers.batch_norm(out)
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 256, [3,3], [1,1], padding='SAME', use_bias=False, name='bnn_conv2d_3')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.contrib.layers.batch_norm(out)
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 512, [3,3], [1,1], padding='SAME', use_bias=False, name='bnn_conv2d_4')
	out = tf.contrib.layers.batch_norm(out)
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 512, [3,3], [1,1], padding='SAME', use_bias=False, name='bnn_conv2d_5')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.contrib.layers.batch_norm(out)
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 1024, use_bias=False, name='binary_dense_1')
	out = tf.contrib.layers.batch_norm(out)
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 1024, use_bias=False, name='binary_dense_2')
	out = tf.contrib.layers.batch_norm(out)
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 10, name='binary_dense_3')
	output = tf.contrib.layers.batch_norm(out)
	
	return input, output
	

def binary_cifar10_sbn(input):
	out = layers.binaryConv2d(input, 128, [3,3], [1,1], padding='VALID', use_bias=False, binarize_input=False, name='bc_conv2d_1')
	out = layers.spatial_shift_batch_norm(out, name='shift_batch_norm_1')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 128, [3,3], [1,1], padding='SAME', use_bias=False, name='bnn_conv2d_1')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = layers.spatial_shift_batch_norm(out, name='shift_batch_norm_2')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 256, [3,3], [1,1], padding='SAME', use_bias=False, name='bnn_conv2d_2')
	out = layers.spatial_shift_batch_norm(out, name='shift_batch_norm_3')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 256, [3,3], [1,1], padding='SAME', use_bias=False, name='bnn_conv2d_3')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = layers.spatial_shift_batch_norm(out, name='shift_batch_norm_4')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 512, [3,3], [1,1], padding='SAME', use_bias=False, name='bnn_conv2d_4')
	out = layers.spatial_shift_batch_norm(out, name='shift_batch_norm_5')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 512, [3,3], [1,1], padding='SAME', use_bias=False, name='bnn_conv2d_5')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = layers.spatial_shift_batch_norm(out, name='shift_batch_norm_6')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 1024, use_bias=False, name='binary_dense_1')
	out = layers.shift_batch_norm(out, name='shift_batch_norm_7')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 1024, use_bias=False, name='binary_dense_2')
	out = layers.shift_batch_norm(out, name='shift_batch_norm_8')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 10, name='binary_dense_3')
	output = layers.shift_batch_norm(out, name='shift_batch_norm_9')
	
	return input, output

	

def mnist(input, training=True):
	out = tf.layers.dense(input, 2048, activation=None)
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.dense(out, 2048, activation=None)
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.dense(out, 2048, activation=None)
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	output = tf.layers.dense(out, 10, activation=None)
	
	return input, output
	

def binary_mnist(input, training=True):
	fc1 = layers.binaryDense(input, 2048, activation=None, name="binarydense1", binarize_input=False)
	bn1 = tf.layers.batch_normalization(fc1, training=training)
	ac1 = tf.clip_by_value(bn1, -1, 1)
	fc2 = layers.binaryDense(ac1, 2048, activation=None, name="binarydense2")
	bn2 = tf.layers.batch_normalization(fc2, training=training)
	ac2 = tf.clip_by_value(bn2, -1, 1)
	fc3 = layers.binaryDense(ac2, 2048, activation=None, name="binarydense3")
	bn3 = tf.layers.batch_normalization(fc3, training=training)
	ac3 = tf.clip_by_value(bn3, -1, 1)
	fc4 = layers.binaryDense(ac3, 10, activation=None, name="binarydense4")
	output =  tf.layers.batch_normalization(fc4, training=training)
	
	return input, output
	
	
def binary_mnist_sbn(input, training=True):
	fc1 = layers.binaryDense(input, 2048, activation=None, name="binarydense1", binarize_input=False)
	bn1 = layers.shift_batch_norm(fc1, training=training, name="batch_norm1")
	ac1 = tf.clip_by_value(bn1, -1, 1)
	fc2 = layers.binaryDense(ac1, 2048, activation=None, name="binarydense2")
	bn2 = layers.shift_batch_norm(fc2, training=training, name="batch_norm2")
	ac2 = tf.clip_by_value(bn2, -1, 1)
	fc3 = layers.binaryDense(ac2, 2048, activation=None, name="binarydense3")
	bn3 = layers.shift_batch_norm(fc3, training=training, name="batch_norm3")
	ac3 = tf.clip_by_value(bn3, -1, 1)
	fc4 = layers.binaryDense(ac3, 10, activation=None, name="binarydense4")
	output = layers.shift_batch_norm(fc4, training=training, name="batch_norm4")
	
	return input, output
	
	
def netlist():
	return ['mlp', 'binary_mlp', 'cifar10', 'binary_cifar10', 'mnist', 'binary_mnist']


def get_network(name, *args, **kargs):
	if(name == 'mlp'):
		return multilayer_perceptron(*args, **kargs)
	elif(name == 'binary_mlp'):
		return binary_multilayer_perceptron(*args, **kargs)
	elif(name == 'cifar10'):
		return cifar100(*args, **kargs)
	elif(name == 'binary_cifar10'):
		return binary_cifar100(*args, **kargs)
	elif(name == 'mnist'):
		return binary_mnist(*args, **kargs)
	elif(name == 'binary_mnist'):
		return binary_mnist(*args, **kargs)
	return None
	

		