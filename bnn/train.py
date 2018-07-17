import tensorflow as tf
import argparse
import networks


parser = argparse.ArgumentParser(description='Beating OpenAI envs with Reinforcement Learning: Training script')

parser.add_argument('--network', dest='network', type=str, default='dqn', choices=networks.netlist(), help='Type of network to be used')
parser.add_argument('--out_folder', dest='out_folder', type=str, default='./model/', help='path where to save network\'s weights')
parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Number of epochs performed during training')
parser.add_argument('--dataset', dest='checkpoint_steps', type=string, help='Dataset path', required=True)

parser.set_defaults(render_training=False)
args = parser.parse_args()


with tf.Session() as sess:
	
	xnet, ynet = multilayer_perceptron([100, 100, 50, 1])
	
	y = tf.placeholder(tf.float32, output_shape)
	
	loss = tf.losses.mean_squared_error(y, ynet)
	opt = self.optimizer.minimize(loss=loss)
	
	for epoch in range(EPOCHS):