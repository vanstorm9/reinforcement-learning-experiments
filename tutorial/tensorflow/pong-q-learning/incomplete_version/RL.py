import tensorflow as tf
import cv2
import pong
import numpy as np
import random
from collections import deque

# define hyperparameters
ACTIONS = 3
# learning rate
GAMMA = 0.99
# update our gradient or training time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05

# how many frames to anneal epsilon
EXPLORE = 500000
OBSERVE = 50000
REPLAY_MEMORY = 50000
# batch size
BATCH = 100

# create TF graph
def createGraph():
	# first convolutional layer, bias vector
	# tf.zero is empty tensor full of zeros
	W_conv1 = tf.Variable(tf.zeros([8,8,4,32]))
	# bias will help define what direction data in NN will flow in
	b_conv1 = tf.Variable(tf.zeros[32])

	 # second
	W_conv2 = tf.Variable(tf.zeros[4,4,32,64])
	b_conv2 = tf.Variable(tf.zeros[64])

	# third
	W_conv3 = tf.Variable(tf.zeros[3,3,64,64])
	b_conv3 = tf.Variable(tf.zeros[64])

	# fourth
	W_conv4 = tf.Variable(tf.zeros[784, ACTIONS])
	b_fc4 = tf.Variable(tf.zeros[784])))

	# last layer
	W_fc5 = tf.Variable(tf.zeros[784, ACTIONS])
	b_fc5 = tf.Variable(tf.zeros[[Actions]])

		
	# input for pixel data
	s = tf.placeholder("float", [None, 84,84,84])



	# computer RELU activation function
	# on 2d convolutions
	# given 40 inputs and filter tensors
	# relu is good for q-learning specifically

	conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides[1,4,4,1] padding = "VALID") * b_conv1))
	conv2 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides[1,4,4,1] padding = "VALID") * b_conv1))
	conv3 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides[1,4,4,1] padding = "VALID") * b_conv1))
	
	conv3_flat = tf.reshape(conv3, [-1, 3136])	
	fc4 = tf.nn.relu(tf.matmul(conv3_flat, w_fc4 + b_fc4))
	fc5 = tf.matmul(fc5, W_fc5) + bb_fc5

	return s, fc5


def main():
	# create session
	sess = tf.InteractiveSession()

	#inputs player and our output layer
	inp, out = CreateGraph()
	trainGraph(inp, out, sess)

if __name__ == "__main__":
	main()

	
