import tensorflow as tf
import layers

lr_bshift = layers.lr_binary_shift


		
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

	
def cifar100(input):
	pass


def binary_cifar100(input):
	pass

def mnist(input):
	dr1 = tf.layer.dropout(input)
	fc1 = tf.layers.dense(dr1, 4096, activation=tf.nn.relu)
	dr2 = tf.layer.dropout(fc1)
	fc2 = tf.layers.dense(fc1, 4096, activation=tf.nn.relu)
	dr3 = tf.layer.dropout(fc1)
	fc3 = tf.layers.dense(fc2, 4096, activation=tf.nn.relu)
	dr4 = tf.layer.dropout(fc1)
	output = tf.softmax(fc4, 32)
	return input, output
	
def binary_mnist(input):
	fc1 = layers.binaryDense(input, 4096, activation=None, name="binarydense1")
	bn1 = tf.contrib.layers.batch_norm(fc1)
	ac1 = tf.clip_by_value(bn1, -1, 1)
	fc2 = layers.binaryDense(ac1, 4096, activation=None, name="binarydense2")
	bn2 = tf.contrib.layers.batch_norm(fc2)
	ac2 = tf.clip_by_value(bn2, -1, 1)
	fc3 = layers.binaryDense(ac2, 4096, activation=None, name="binarydense3")
	bn3 = tf.contrib.layers.batch_norm(fc3)
	ac3 = tf.clip_by_value(bn3, -1, 1)
	fc4 = layers.binaryDense(ac3, 4096, activation=None, name="binarydense4")
	bn4 = tf.contrib.layers.batch_norm(fc4)
	output = tf.softmax(bn4, 32)
	
	return input, output
	
	
def netlist():
	return ['mlp', 'binary_mlp', 'cifar100', 'binary_cifar100', 'mnist', 'binary_mnist']


def get_network(name, *args, **kargs):
	if(name == 'mlp'):
		return multilayer_perceptron(*args, **kargs)
	elif(name == 'binary_mlp'):
		return binary_multilayer_perceptron(*args, **kargs)
	elif(name == 'cifar100'):
		return cifar100(*args, **kargs)
	elif(name == 'binary_cifar100'):
		return binary_cifar100(*args, **kargs)
	elif(name == 'mnist'):
		return binary_mnist(*args, **kargs)
	elif(name == 'binary_mnist'):
		return binary_mnist(*args, **kargs)
	return None
	

		