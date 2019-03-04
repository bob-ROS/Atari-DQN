import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt 
from collections import deque
from keras.utils import plot_model
import random
from PIL import Image
import torchvision.transforms as transf
import torch
import tensorflow as tf

np.random.seed(1234)


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

def observationProcessing(state, stacked_frames,new_episode = False ):
    state = state[30:200,5:155,:]
    state = transform(state)
    state = np.array(state).reshape((1, 84, 84))

    if new_episode:
        #stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(4)], maxlen=4)
        stacked_frames.append(state)
        stacked_frames.append(state)
        #stacked_frames.append(state)
        #stacked_frames.append(state)

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
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.gamma = 0.90
        self.memory = deque(maxlen=100000)
        self.learning_rate = 0.00025
        self.batch_size = 32
        self.model = self._create_model()


    def _create_model(self):
        model = Sequential()
        #model.add(Convolution2D(input_shape = (200, 600, 1), filters=32,kernel_size=8,strides=(4,4),padding='valid', data_format='channels_last', kernel_initializer='zeros',bias_initializer='zeros',activation='relu'))
        #model.add(Convolution2D(filters=64,kernel_size=4,strides=(2,2),padding='valid', data_format='channels_last',kernel_initializer='zeros', bias_initializer='zeros', activation='relu'))
        #model.add(Convolution2D(filters=64, kernel_size=4, strides=(1, 1), padding='valid', kernel_initializer='zeros',bias_initializer='zeros', data_format='channels_last', activation='relu'))
        #model.add(Flatten())
        #model.add(Dense(units=512,activation='relu',kernel_initializer='zeros', bias_initializer='zeros'))
        #model.add(Dense(units=self.act_size,activation='linear',kernel_initializer='he_uniform', bias_initializer='zeros'))
        model.add(Convolution2D(32, (8, 8), subsample=(4, 4),  activation='relu',input_shape=(84, 84, 2)))
        #model.add(BatchNormalization())
        model.add(Convolution2D(64, (4, 4), strides=(2, 2) ,activation='relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1) ,activation='relu'))
        model.add(Flatten())
        model.add(Dense(126, activation='relu'))
        model.add(Dense(self.act_size,  activation='linear'))
        model.compile(loss=huber_loss, optimizer=Adam(lr=self.learning_rate))
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
            q_value = 0 # assume punishment
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
    env = gym.make('BreakoutDeterministic-v4')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    transform = transf.Compose([transf.ToPILImage(), transf.Resize((84,84)), transf.Grayscale(1)])

    DQN = DQN(state_size,action_size)
    plot_model(DQN.model, to_file='model.png',  show_layer_names=True, show_shapes=True)
    episodes = 10000000
    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(2)], maxlen=2)
    rew_max = 0
    for i in range(episodes):
        new_episode = True
        state = env.reset()
        stacked_frames, state = observationProcessing(state, stacked_frames,new_episode)
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
            stacked_frames, next_state = observationProcessing(next_state, stacked_frames,new_episode)
            
            if not done:
                DQN._append_mem(state, action, reward, next_state, done)
            else:
                #print "reward: {}".format(reward)
                #reward = -reward #punishment sufficient?
                next_state = 0
                if tot_rew > rew_max:
                    rew_max = tot_rew
                DQN._append_mem(state, action, reward, next_state, done)
                if len(DQN.memory) >= DQN.batch_size:
                    DQN._train()
                print("episode: {}/{}, score: {}, epsilon: {:.2}, max score: {}".format(i, episodes, tot_rew, DQN.epsilon,rew_max))

            time += 1
            state = next_state

