import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split


def load_mnist():
	mnist = input_data.read_data_sets('dataset/MNIST_data', one_hot=True)
	x_train = mnist.train.images
	y_train = mnist.train.labels
	x_test = mnist.test.images
	y_test = mnist.test.labels
	return x_train, y_train, x_test, y_test
	


def random_dataset():
	n_samples = 1000
	idim, odim = 30, 3
	x = np.random.rand(n_samples, idim)
	y = np.zeros((n_samples, odim))
	J = np.random.choice(odim, n_samples)
	y[np.arange(n_samples), J] = 1
	
	x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=0.3, random_state=42)
	return x_trn, y_trn, x_tst, y_tst