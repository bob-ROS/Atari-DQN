import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt 
from collections import deque
import random


#unprocessed_image_size = fromgame

#preprocessed_image_size = x


# return observationProcessed(imagefromstep)
def observationProcessing(env):
    env.reset()
    screen = env.render(mode='rgb_array')
    #screen = np.mean(screen,-1)
    screen = screen.mean(-1)
    #screen = screen[0:350][150:450]
    screen = screen[150:350][:]
    plt.imshow(screen, cmap = plt.get_cmap('gray'))
    plt.show()
    print screen.shape

    return screen
#initialize convolutional network


#save / load network


#experience_replay


# MAIN

#loop
	#add.parser visualize or not

	#blah = gym.step()
	#obsevationProcessed(blah)

	#experience_replay


#if __name__ == "__main__":

class DQN:
    def __init__(self, state_size,act_size):
        self.state_size = state_size
        self.act_size = act_size
        self.input_dim = (200,600,1)
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.0001
        self.model = self._create_model()


    def _create_model(self):
        model = Sequential()
        model.add(Convolution2D(input_shape=self.input_dim, filters=32,kernel_size=[8,8],strides=[4,4],padding='valid', data_format='channels_last', kernel_initializer='glorot_uniform', activation='elu'))
        model.add(Convolution2D(filters=64,kernel_size=[4,4],strides=[2,2],padding='valid', kernel_initializer='glorot_uniform', activation='elu'))
        model.add(Flatten())
        model.add(Dense(units=512,activation='elu',kernel_initializer='glorot_uniform'))
        model.add(Dense(units=self.act_size,activation=None,kernel_initializer='glorot_uniform'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _append_mem(self, state, action, reward, next_state):
        self.memory.append((state,action,reward,next_state))

    def _predict(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.act_size)
        act_values = self.model.predict(state)
        action = np.argmax(act_values)
        return action



if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    observationProcessing(env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    DQN = DQN(state_size,action_size)
    episodes = 5
    for i in range(episodes):

        state = observationProcessing(env)

        action = DQN._predict(state)
        next_state, reward, done, lives = env.step(action)
        while not done:
            #experience replay

            DQN._append_mem(state, action, reward, next_state)
            state = next_state

        if done:
            print "u fucking cunt"
