import tensorflow as tf
import layers


# class network():
	
	# def __init__(self, *args, **kargs):
		# self.input, self.output = _architecture(*args, **kargs)
		
	# def _architecture(*args, **kargs):
		# return None, None
		
	
def multilayer_perceptron(units_list):
	input = tf.placeholder(tf.float32, [None, units_list[0]])
	output = input
	for l in range(1, len(units_list)):
		output = tf.layers.dense(output, units_list[l], activation=tf.nn.relu)
		
	return input, output
	
	
def binary_multilayer_perceptron(units_list):
	input = tf.placeholder(tf.float32, [None, units_list[0]])
	output = input
	for l in range(1, len(units_list)):
		output = layers.binaryDense(output, units_list[l], activation=tf.nn.relu)
		
	return input, output

	
def cifar100():
	pass


def binary_cifar100():
	pass

def mnist():
	pass
	
def binary_mnist():
	pass
	
	
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
	

		