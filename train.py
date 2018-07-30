import tensorflow as tf
import numpy as np
import argparse
import networks
import math
import time
import os

import utils.datasets as datasets
from utils.progressbar import ProgressBar




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

session_logdir = os.path.join(LOGDIR, str(timestamp))
train_logdir = os.path.join(session_logdir, 'train')
test_logdir = os.path.join(LOGDIR, str(timestamp), 'test')

if not os.path.exists(MODELDIR):
	os.mkdir(MODELDIR)
if not os.path.exists(train_logdir):
	os.makedirs(train_logdir)
if not os.path.exists(test_logdir):
	os.makedirs(test_logdir)
	

	
# dataset preparation using tensorflow dataset iterators
x_train, y_train, x_test, y_test = datasets.load_mnist() #random_dataset()

batch_size = tf.placeholder(tf.int64)
data_features, data_labels = tf.placeholder(tf.float32, (None,)+x_train.shape[1:]), tf.placeholder(tf.float32, (None,)+y_train.shape[1:])

train_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
train_data = train_data.batch(batch_size).repeat().shuffle(x_train.shape[0])

test_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
test_data = test_data.batch(batch_size).repeat().shuffle(x_train.shape[0])

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
	
	global_step = tf.train.get_or_create_global_step()
	train_op = optimizer.minimize(loss=cross_entropy, global_step=global_step)

# metrics definition
with tf.variable_scope('metrics'):
	mloss, mloss_update	  = tf.metrics.mean(loss)
	accuracy, acc_update  = tf.metrics.accuracy(  tf.argmax(labels, axis=1), tf.argmax(ysoft, axis=1))

	metrics = [mloss, accuracy]
	metrics_update = [mloss_update, acc_update]

# Isolate the variables stored behind the scenes by the metric operation
metrics_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
metrics_initializer = tf.variables_initializer(metrics_variables)


# summaries
los_sum = tf.summary.scalar('loss', mloss)
acc_sum = tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge([los_sum, acc_sum])


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
		
		print("\nEPOCH %d/%d" % (epoch+1, EPOCHS))
		
		
		# initialize training dataset
		sess.run(train_initialization, feed_dict={data_features:x_train, data_labels:y_train, batch_size:BATCH_SIZE})
		sess.run(metrics_initializer)
		
		progress_info = ProgressBar(total=NUM_BATCHES_TRAIN, prefix=' train', show=True)
		
		# Training of the network
		for nb in range(NUM_BATCHES_TRAIN):
			outg, outn, _= sess.run([labels, ysoft, train_op])	# train network on a single batch
			batch_trn_loss, _ = sess.run(metrics_update)
			trn_loss, a = sess.run(metrics)
			
			progress_info.update_and_show( suffix = '  loss {:.4f},  acc: {:.3f}'.format(trn_loss, a) )
		print()
		
		summary = sess.run(merged_summary)
		train_writer.add_summary(summary, epoch)
		
		
		
		# initialize the test dataset
		sess.run(test_initialization, feed_dict={data_features:x_test, data_labels:y_test, batch_size:BATCH_SIZE})
		sess.run(metrics_initializer)
		
		progress_info = ProgressBar(total=NUM_BATCHES_TEST, prefix='  eval', show=True)
		
		# evaluation of the network
		for nb in range(NUM_BATCHES_TEST):
			sess.run([loss, metrics_update])
			val_loss, a = sess.run(metrics)
			
			progress_info.update_and_show( suffix = '  loss {:.4f},  acc: {:.3f}'.format(val_loss, a) )
		print()
		
		summary  = sess.run(merged_summary)
		test_writer.add_summary(summary, epoch)
		
	
	train_writer.close()
	test_writer.close()

	saver.save(sess, MODELDIR+"/model.ckpt")

print('\nTraining completed!\nNetwork model is saved in  {}\nTraining logs are saved in {}'.format(MODELDIR, session_logdir))