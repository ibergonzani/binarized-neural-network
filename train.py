import tensorflow as tf
import numpy as np
import argparse
import networks
import math
import time
import os

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def load_mnist():
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	x_train = mnist.train.images
	y_train = mnist.train.labels
	x_test = mnist.test.images
	y_test = mnist.test.labels
	return x_train, y_train, x_test, y_test
	

def random_dataset():
	n_samples = 1000
	idim = 30
	odim = 3
	x = np.random.rand(n_samples, idim)
	#y = np.array([ [(np.sign(np.sin(x1*x2)*x3)+1)/2, np.around(x1)] for (x1, x2, x3) in x])
	y = np.zeros((n_samples, odim))
	J = np.random.choice(odim, n_samples)
	y[np.arange(n_samples), J] = 1
	
	x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=0.3, random_state=42)
	return x_trn, y_trn, x_tst, y_tst
		

parser = argparse.ArgumentParser(description='Training module for binarized nets')
parser.add_argument('--network', dest='network', type=str, choices=networks.netlist(), help='Type of network to be used')
parser.add_argument('--modeldir', dest='modeldir', type=str, default='./models/', help='path where to save network\'s weights')
parser.add_argument('--logdir', dest='logdir', type=str, default='./logs/', help='folder for tensorboard logs')
parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Number of epochs performed during training')
parser.add_argument('--batchsize', dest='batchsize', type=int, default=32, help='Dimension of the training batch')
args = parser.parse_args()

MODELDIR = args.modeldir
LOGDIR = args.logdir
EPOCHS = args.epochs
BATCH_SIZE = args.batchsize


timestamp = int(time.time())

train_logdir = os.path.join(LOGDIR, str(timestamp), 'train')
test_logdir = os.path.join(LOGDIR, str(timestamp), 'test')

if not os.path.exists(MODELDIR):
	os.mkdir(MODELDIR)
if not os.path.exists(train_logdir):
	os.makedirs(train_logdir)
if not os.path.exists(test_logdir):
	os.makedirs(test_logdir)
	

# dataset preparation using tensorflow dataset iterators
x_train, y_train, x_test, y_test = load_mnist() #random_dataset()

batch_size = tf.placeholder(tf.int64)
data_features, data_labels = tf.placeholder(tf.float32, (None,)+x_train.shape[1:]), tf.placeholder(tf.float32, (None,)+y_train.shape[1:])

train_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
train_data = train_data.batch(batch_size).repeat()

test_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
test_data = test_data.batch(batch_size).repeat()

data_iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

features, labels = data_iterator.get_next()
train_initialization = data_iterator.make_initializer(train_data)
test_initialization = data_iterator.make_initializer(test_data)

# network initialization
xnet, ynet = networks.multilayer_perceptron(features, [2048, 2048, 2048, 10])
ysoft = tf.nn.softmax(ynet)

with tf.name_scope('trainer_optimizer'):
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
	loss = tf.losses.mean_squared_error(labels, ysoft)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ynet, labels=tf.argmax(labels, axis=1))
	train_op = optimizer.minimize(loss=cross_entropy)

with tf.name_scope('metrics'):
	accuracy, acc_update = tf.metrics.accuracy(tf.argmax(labels, axis=1), tf.argmax(ysoft, axis=1), name="metric")
	precision, pre_update = tf.metrics.precision(tf.argmax(labels, axis=1), tf.argmax(ysoft, axis=1), name="metric")
	recall, rec_update = tf.metrics.recall(tf.argmax(labels, axis=1), tf.argmax(ysoft, axis=1), name="metric")

	metrics = [accuracy, precision, recall]
	metrics_update = [acc_update, pre_update, rec_update]

# Isolate the variables stored behind the scenes by the metric operation
running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metric")
metrics_initializer = tf.variables_initializer(var_list=running_vars)
	

# network weights saver
saver = tf.train.Saver()

NUM_BATCHES_TRAIN = math.ceil(x_train.shape[0] / BATCH_SIZE)
NUM_BATCHES_TEST = math.ceil(x_test.shape[0] / BATCH_SIZE)

with tf.Session() as sess:

	# tensorboard summary writer
	train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
	test_writer = tf.summary.FileWriter(test_logdir)
	
	sess.run(tf.global_variables_initializer())
	
	for epoch in range(EPOCHS):
		
		trn_loss = 0
		val_loss = 0
		
		# initialize training dataset
		sess.run(train_initialization, feed_dict={data_features:x_train, data_labels:y_train, batch_size:BATCH_SIZE})
		sess.run(metrics_initializer)
		
		# Training of the network
		for nb in range(NUM_BATCHES_TRAIN):
			outg, outn, batch_trn_loss, _, _, _, _ = sess.run([labels, ysoft, loss, train_op] + metrics_update)	# train network on a single batch
			a, p, r = sess.run(metrics)
			
			trn_loss = trn_loss + batch_trn_loss											# accumulating loss
			print("EPOCH %d, training %.1f%%,   acc: %.3f   prec: %.3f   rec: %.3f   sk %.3f"
					% (epoch+1, 100*(nb+1)/NUM_BATCHES_TRAIN, a, p, r, accuracy_score(outg, np.around(outn))), end='\r')
		
		trn_loss = trn_loss / NUM_BATCHES_TRAIN				# naive mean loss
		print("EPOCH %d, training 100%%, loss %.4f,  acc: %.3f   prec: %.3f   rec: %.3f"
				% (epoch+1, trn_loss, a, p, r))
			
		
		# initialize the test dataset
		sess.run(test_initialization, feed_dict={data_features:x_test, data_labels:y_test, batch_size:BATCH_SIZE})
		sess.run(metrics_initializer)
		
		# evaluation of the network
		for nb in range(NUM_BATCHES_TEST):
			batch_val_loss = sess.run(loss)				# evaluate network on single batch
			val_loss = val_loss + batch_val_loss		# accumulating loss
			print("EPOCH %d, evaluation %.1f%%" % (epoch+1, 100*(nb+1)/NUM_BATCHES_TEST), end='\r')
		
		val_loss = val_loss / NUM_BATCHES_TEST			# naive mean loss
		print("EPOCH %d, evaluation 100%%, loss %.4f" % (epoch+1, val_loss))
		
		
		summary = tf.Summary(value=[tf.Summary.Value(tag="MSE loss", simple_value=trn_loss)])
		train_writer.add_summary(summary, epoch)
		summary = tf.Summary(value=[tf.Summary.Value(tag="MSE loss", simple_value=val_loss)])
		test_writer.add_summary(summary, epoch)
	
	
	train_writer.close()
	test_writer.close()

	saver.save(sess, MODELDIR+"/model.ckpt")
