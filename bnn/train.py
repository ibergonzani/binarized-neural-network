import tensorflow as tf
import argparse
import networks
import os

from tensorflow.examples.tutorials.mnist import dataset
from sklearn.model_selection import KFold


class Trainer():
	def __init__(net_input, net_output, optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)):
		self.net_input = net_input
		self.net_output = net_output
		self.optimizer = optimizer
		
		self.target = tf.placeholder(tf.float32, net_output.get_shape())
	
		self.loss = tf.losses.mean_squared_error(y, ynet)
		self.opt = self.optimizer.minimize(loss=loss)
		
	def train(sess, x_batch, y_batch):
		loss, _ = sess.run([self.loss, self.opt], feed_dict={self.net_input: x_batch, self.target: y_batch})
		return loss
	
	def eval(sess, x_batch, y_batch):
		y_net, loss = sess.run([self.net_output, self.loss], feed_dict={self.net_input: x_batch})
		return loss


def load_mnist():
	mnist = dataset.read_data_sets("data/mnist-tf", one_hot=True)
	x_train = mnist.train.images
	y_train = mnist.train.labels
	x_test = mnist.test.images
	y_test = mnist.test.labels
	return x_train, y_train, x_test, y_test

	
def cross_validation(sess, splits, trainer, x_train, y_train):
	
	trn_loss = 0
	val_loss = 0
	
	kf = KFold(n_splits=splits)
	for trn_ids, val_ids in kf.split(x_train, y_train):
		x_trn, y_trn = x_train[trn_ids], y_train[trn_ids]
		x_val, y_val = x_train[val_ids], y_train[val_ids]
		
		fold_trn_loss = trainer.train(sess, x_trn, y_trn)
		fold_val_loss = trainer.eval(sess, x_val, y_val)
		
		trn_loss = trn_loss + fold_trn_loss
		val_loss = val_loss + fold_val_loss
	
	trn_loss = trn_loss / splits
	val_loss = val_loss / splits
	
	return trn_loss, val_loss
		

parser = argparse.ArgumentParser(description='Beating OpenAI envs with Reinforcement Learning: Training script')

parser.add_argument('--network', dest='network', type=str, default='dqn', choices=networks.netlist(), help='Type of network to be used')
parser.add_argument('--out_folder', dest='out_folder', type=str, default='./model/', help='path where to save network\'s weights')
parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Number of epochs performed during training')
parser.add_argument('--dataset', dest='dataset_path', type=string, help='Dataset path', required=True)

args = parser.parse_args()

use_crossvalidation = True

with tf.Session() as sess:
	
	xnet, ynet = multilayer_perceptron([100, 100, 50, 1])
	trainer = Trainer(xnet, ynet)
	
	x_train, y_train, x_test, y_test = load_mnist():
	
	for epoch in range(EPOCHS):
		
		if use_crossvalidation:
			trn_loss, val_loss = cross_validation(sess, 10, trainer, x_train, y_train)
		else:
			trn_loss = 0
			for (x_batch, y_batch) in np.split(zip(x_train, y_train), 10):
				trn_batch_loss = trainer.train(sess, x_batch, y_batch)
				trn_loss = trn_loss + trn_batch_loss
			trn_loss = trn_loss / 10
		
		writer.add_summary(trn_loss, epoch)
		writer.add_summary(val_loss, epoch)
		
		