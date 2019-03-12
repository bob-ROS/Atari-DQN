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

np.random.seed(1234)

def observationProcessing(state, stacked_frames,new_episode = False ):
    proc_state = transform(state)
    proc_state = np.array(proc_state)
    state = proc_state[17:97,:]
    state = np.array(state).reshape((1, 80, 80))

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


    #double_state = stacked_state[80,80,4] - stacked_state[80,80,3]
    return stacked_frames, stacked_state #1*80*80*4

class PreProcessing:
    def __init__(self):
        self.past_frames = deque([np.zeros((80, 80), dtype=np.uint8) for i in range(1)], maxlen=4)
        self.transform = transf.Compose([transf.ToPILImage(), transf.Resize((105,80)), transf.Grayscale(1)])

    def rescale_crop(self,frame):
         img = self.transform(frame)
         img = np.array(img)
         proc_img = img[17:97,:]
         proc_img = np.array(proc_img).reshape((1, 80, 80))
         return proc_img

    def proc(self,frame,new_ep):
        proc_image = self.rescale_crop(frame)
        self.past_frames.append(proc_image)
        if len(self.past_frames) == 4:
            motion = (self.past_frames[0] + self.past_frames[1]) - (self.past_frames[3]+self.past_frames[2])
        else:
            motion = proc_image
        state = np.stack([proc_image, motion], axis=-1)
        return state





class DQN:
    def __init__(self, state_size,act_size):
        self.state_size = state_size
        self.act_size = act_size
        self.input_dim = [0,200,600,1]
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.gamma = 0.90
        self.memory = deque(maxlen=20000)
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self._create_model()


    def _create_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (8, 8), strides=(4, 4),  activation='relu',input_shape=(80, 80, 2)))
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

            q_table = self.model.predict(state)

            q_table[0][action] = q_value
            self.model.fit(state, q_table, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



if __name__ == "__main__":
    env = gym.make('Breakout-v4')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    DQN = DQN(state_size,action_size)
    episodes = 10000000
    #stacked_frames = deque([np.zeros((80, 80), dtype=np.uint8) for i in range(2)], maxlen=4)
    pre = PreProcessing()
    rew_max = 0
    for i in range(episodes):
        new_episode = True
        state = env.reset()
        state = pre.proc(state, new_episode)
        new_episode=False
        done = False
        time = 0
        tot_rew=0
        while not done:
            #state = observationProcessing(env)
            env.render()
            action = DQN._predict(state)
            #print DQN.model.predict(state)

            next_state, reward, done, _ = env.step(action)
            #experience replay
            tot_rew += reward
            #last_state = currystate
            next_state = pre.proc(next_state,new_episode)
            
            if not done:
                DQN._append_mem(state, action, reward, next_state, done)
            else:
                next_state = 0
                if tot_rew > rew_max:
                    rew_max = tot_rew
                DQN._append_mem(state, action, reward, next_state, done)
                print("episode: {}/{}, score: {}, epsilon: {:.2}, max score: {}".format(i, episodes, tot_rew, DQN.epsilon,rew_max))

            time += 1
            state = next_state
            if len(DQN.memory) == 1500:
                for i in range(100):
                    if len(DQN.memory) >= DQN.batch_size:
                        DQN._train()


