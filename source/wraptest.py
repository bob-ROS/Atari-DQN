import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt 
from collections import deque
from keras.utils import plot_model
import random
from PIL import Image
import torchvision.transforms as transf
import torch
import wrappers as wr

np.random.seed(1234)

def observationProcessing(state, stacked_frames,new_episode = False ):
    state = state[30:200,5:155,:]
    state = transform(state)
    state = np.array(state).reshape((1, 84, 84))

    if new_episode:
        #stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(4)], maxlen=4)
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)

        stacked_state = np.stack(stacked_frames, axis=-1)


    else:
        stacked_frames.append(state)
        stacked_state = np.stack(stacked_frames, axis=-1)

    #In Keras, need to reshape



    return stacked_frames, stacked_state #1*200*600*4




class DQN:
    def __init__(self, state_size,act_size):
        self.state_size = state_size
        self.act_size = act_size
        self.input_dim = [0,200,600,1]
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.02
        self.gamma = 0.99
        self.memory = deque(maxlen=100000)
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self._create_model()


    def _create_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (8, 8), subsample=(4, 4),  activation='relu',input_shape=(84, 84, 4)))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2) ,activation='relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1) ,activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.act_size,  activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _append_mem(self, state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state, done))

    def _predict(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.act_size)
        act_values = self.model.predict(state)
        action = np.argmax(act_values)
        #print act_values
        return action

    def _train(self):
        minibatch = random.sample(self.memory, self.batch_size)
        #minibatch = self.memory
        for state, action, reward, next_state, done in minibatch:
            q_value = reward # assume punishment
            if not done:
                q_value = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            #print "wtf1"
            #target = q_value
            #print self.model.predict(state)
            q_table = self.model.predict(state)
            #print q_value
            #print q_table[0]
            q_table[0][action] = q_value
            self.model.fit(state, q_table, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            #print "wtf2"
            self.epsilon *= self.epsilon_decay



if __name__ == "__main__":
    #env = gym.make('Breakout-v4')

    env = wr.make_atari('BreakoutNoFrameskip-v4')
    env = wr.wrap_deepmind(env, frame_stack=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    #transform = transf.Compose([transf.ToPILImage(), transf.Resize((84,84)), transf.Grayscale(1)])

    DQN = DQN(state_size,action_size)
    #plot_model(DQN.model, to_file='model.png',  show_layer_names=True, show_shapes=True)
    episodes = 1000000
    #stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(4)], maxlen=4)
    rew_max = 0
    tot_rew=0
    ep_rew=0
    avg_100rew=[] 
    for i in range(episodes):
        new_episode = True
        state = np.array(env.reset()).reshape((1, 84, 84,4))
        #stacked_frames, state = observationProcessing(state, stacked_frames,new_episode)
        new_episode=False
        done = False
        while not done:

            env.render()
            action = DQN._predict(state)


            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state).reshape((1, 84, 84,4))
            lives = info['ale.lives']
            tot_rew += reward
            
            if not done:
                DQN._append_mem(state, action, reward, next_state, done)
                
            if done:
                if tot_rew > rew_max:
                    rew_max = tot_rew
                DQN._append_mem(state, action, reward, next_state, done)
                print("episode: {}/{}, score: {}, epsilon: {:.2}, max score: {}".format(i, episodes, tot_rew, DQN.epsilon,rew_max))
                avg_100rew.append(tot_rew)
                if i%100 ==0:
                    avg100 = sum(avg_100rew)/len(avg_100rew)
                    del avg_100rew[:]
                    with open("Output.txt", "a") as text_file:
                        text_file.write("episode: {}/{}, reward: {}, epsilon: {:.2}, max reward: {}, mean past 100 rewards: {}\n".format(i, episodes, tot_rew, DQN.epsilon,rew_max, avg100))
            if lives == 0:
                tot_rew=0
            if len(DQN.memory) >= DQN.batch_size:
                DQN._train()
            state = next_state


