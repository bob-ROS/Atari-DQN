import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt 
from collections import deque
import random


class DQN:
	def __init__(self, state_size,act_size):
		self.state_size = state_size
		self.act_size = act_size
		self.memory = deque(maxlen=2000)
		self.model = self._create_model()

	def _create_model(self):
		model = Sequential()
		model.add(Convolution2D(input_shape=input_dim, filters=32,kernel_size=[8,8],strides=[4,4],padding='valid', data_format='channels_last', kernel_initializer='glorot_uniform', activation='elu'))
		model.add(Convolution2D(filters=64,kernel_size=[4,4],strides=[2,2],padding='valid', kernel_initializer='glorot_uniform', activation='elu'))
		model.add(Flatten())	
		model.add(Dense(units=512,activation='elu',kernel_initializer='glorot_uniform'))
		model.add(Dense(units=self.act_size,activation=None,kernel_initializer='glorot_uniform'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		return model

	def _append_mem(self, state, action, reward, next_state):
		self.memory.append((state,action,reward,next_state))



if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_size.n

	DQN = DQN(state_size,action_size)

	for i in range(episodes):
		env.reset()
		
