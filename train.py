import tensorflow as tf
import numpy as np
import argparse
import networks
import math
import time
import os

import utils.datasets as datasets
from utils.progressbar import ProgressBar
import optimizers



parser = argparse.ArgumentParser(description='Training module for binarized nets')
parser.add_argument('--network', type=str, default='standard', choices=['standard','binary','binary_sbn'], help='Type of network to be used')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist','cifar10'], help='Dataset to be used for the learning task')
parser.add_argument('--modeldir', type=str, default='./models/', help='path where to save network\'s weights')
parser.add_argument('--logdir', type=str, default='./logs/', help='folder for tensorboard logs')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs performed during training')
parser.add_argument('--batchsize', type=int, default=32, help='Dimension of the training batch')
parser.add_argument('--stepsize', type=float, default=1e-3, help='Starting optimizer learning rate value')
parser.add_argument('--shift_optimizer', default=False, action='store_true', help='Toggle th use of shift based AdaMax instead of vanilla Adam optimizer')
args = parser.parse_args()

NETWORK = args.network
DATASET = args.dataset
MODELDIR = args.modeldir
LOGDIR = args.logdir
EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
STEPSIZE = args.stepsize
SHIFT_OPT = args.shift_optimizer


timestamp = int(time.time())

model_name = ''.join([str(timestamp), '_', NETWORK, '_', DATASET])
session_logdir = os.path.join(LOGDIR, model_name)
train_logdir = os.path.join(session_logdir, 'train')
test_logdir = os.path.join(session_logdir, 'test')
session_modeldir = os.path.join(MODELDIR, model_name)

if not os.path.exists(session_modeldir):
	os.makedirs(session_modeldir)
if not os.path.exists(train_logdir):
	os.makedirs(train_logdir)
if not os.path.exists(test_logdir):
	os.makedirs(test_logdir)
	

	
# dataset preparation using tensorflow dataset iterators
x_train, y_train, x_test, y_test, num_classes = datasets.load_dataset(DATASET)

batch_size = tf.placeholder(tf.int64)
data_features, data_labels = tf.placeholder(tf.float32, (None,)+x_train.shape[1:]), tf.placeholder(tf.int32, (None,)+y_train.shape[1:])

train_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
train_data = train_data.batch(batch_size).repeat().shuffle(x_train.shape[0])

test_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
test_data = test_data.batch(batch_size).repeat().shuffle(x_train.shape[0])

data_iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

features, labels = data_iterator.get_next()
train_initialization = data_iterator.make_initializer(train_data)
test_initialization = data_iterator.make_initializer(test_data)


# network initialization
is_training = tf.get_variable('is_training', initializer=tf.constant(False, tf.bool))
switch_training_inference = tf.assign(is_training, tf.logical_not(is_training))

xnet, ynet = networks.get_network(NETWORK, DATASET, features, training=is_training)
ysoft = tf.nn.softmax(ynet)

with tf.name_scope('trainer_optimizer'):
	learning_rate = tf.Variable(STEPSIZE, name='learning_rate')
	learning_rate_decay = tf.placeholder(tf.float32, shape=(), name='lr_decay')
	update_learning_rate = tf.assign(learning_rate, learning_rate / learning_rate_decay)
	
	opt_constructor = optimizers.ShiftBasedAdaMaxOptimizer if SHIFT_OPT else tf.train.AdamOptimizer
	print(opt_constructor)
	optimizer = opt_constructor(learning_rate=learning_rate)
	loss = tf.square(tf.losses.hinge_loss(tf.one_hot(labels, num_classes), ysoft))
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ynet, labels=labels)
	
	global_step = tf.train.get_or_create_global_step()
	train_op = optimizer.minimize(loss=cross_entropy, global_step=global_step)

	
# metrics definition
with tf.variable_scope('metrics'):
	mloss, mloss_update	  = tf.metrics.mean(loss)
	accuracy, acc_update  = tf.metrics.accuracy(labels, tf.argmax(ysoft, axis=1))

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
		
		# exponential learning rate decay
		if (epoch + 1) % 10 == 0:
			sess.run(update_learning_rate, feed_dict={learning_rate_decay: 2.0})
		
		
		# initialize training dataset and set batch normalization training
		sess.run(train_initialization, feed_dict={data_features:x_train, data_labels:y_train, batch_size:BATCH_SIZE})
		sess.run(metrics_initializer)
		sess.run(switch_training_inference)
		
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
		
		
		
		# initialize the test dataset and set batc normalization inference
		sess.run(test_initialization, feed_dict={data_features:x_test, data_labels:y_test, batch_size:BATCH_SIZE})
		sess.run(metrics_initializer)
		sess.run(switch_training_inference)
		
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
	
	saver.save(sess, os.path.join(session_modeldir, 'model.ckpt'))

print('\nTraining completed!\nNetwork model is saved in  {}\nTraining logs are saved in {}'.format(session_modeldir, session_logdir))