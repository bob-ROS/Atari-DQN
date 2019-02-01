from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
import numpy as np
import gym
import time
import random


class Agent():
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=1000)
		self.gamma = 0.95
		self.epsilon = 0.01
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.0001
		self.model = self._create_model()

	def _create_model(self):
		model = Sequential()
		model.add(Dense(128, input_dim=self.state_size, activation='relu'))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		return model


	def remember(self, state, action, reward, next_state, done):
   		self.memory.append((state, action, reward, next_state, done))

	def replay(self, batch_size):
		#minibatch = random.sample(self.memory, batch_size)
		minibatch = self.memory
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
				target_f = self.model.predict(state)
				target_f[0][action] = target
				self.model.fit(state, target_f, epochs=1, verbose=0)
				#if self.epsilon > self.epsilon_min:
					#self.epsilon *= self.epsilon_decay

	def act(self, state):
		if np.random.random() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action


if __name__ == "__main__":

	batch_size = 1000
	episodes          = 100000
	env               = gym.make('BeamRider-ramDeterministic-v4')
	#env               = gym.make('CartPole-v1')
	state_size        = env.observation_space.shape[0]
	action_size       = env.action_space.n
	agent = Agent(state_size, action_size)
	done = False
	rewardtot = 0

	for i in range(episodes):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		#print(i)
		#for time in range(10000):
		previouslives = 3
		while 1:
			env.render()
			#env.spec("")
			action = agent.act(state)
			next_state, reward, done, lives = env.step(action)

			#print(lives['ale.lives'])
			rewardtot += reward
			reward = reward if previouslives == lives['ale.lives']  else -100
			#reward = reward if not done else -50
			next_state = np.reshape(next_state, [1, state_size])
			previouslives =  lives['ale.lives']
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			#print len(agent.memory)
			if done:
				print("episode: {}/{}, score: {}, i: {:.2}".format(i, episodes, rewardtot, agent.epsilon))
				rewardtot = 0
				break
			if len(agent.memory) > batch_size-1:
				agent.replay(batch_size)
				agent.memory.clear()