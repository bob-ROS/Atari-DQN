import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D
from keras.optimizers import Adam
from keras import optimizers
import matplotlib.pyplot as plt 
from collections import deque
from keras.utils import plot_model
import random
from PIL import Image
import torchvision.transforms as transf
import torch
import wrappers as wr

#np.random.seed(1234)

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
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.gamma = 0.90
        self.memory = deque(maxlen=200000)
        self.learning_rate = 0.001
        self.batch_size = 128
        self.model = self._create_model()
        self.target_net = self._create_model()
        self.target_net.set_weights(self.model.get_weights())


    def _create_model(self):
        model = Sequential()
        model.add(Convolution2D(16, (8, 8), strides=(4, 4),  activation='relu',input_shape=(80, 80, 2)))
        model.add(Convolution2D(32, (4, 4), strides=(2, 2) ,activation='relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1) ,activation='relu'))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(self.act_size,  activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0))
        try:
            model.load_weights('my_weights.model')
            print("Succeeded to load model")
        except:
            print("Failed to load model")

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
        minloss = 0.015;
        previousdata= deque(maxlen=100)
        #Do something with mean
        while True:
            minibatch = random.sample(self.memory, self.batch_size)
            #minibatch = self.memory
            state =  np.array([each[0] for each in minibatch], ndmin=4)
            action = np.array([each[1] for each in minibatch])
            reward = np.array([each[2] for each in minibatch])
            next_state = np.array([each[3] for each in minibatch], ndmin=4)
            done = np.array([each[4] for each in minibatch])
            q_value =[]

            for i in range(0, len(minibatch)):
                if done[i] == True:
                    q_value.append(reward[i])
                else:
                  q_value.append(reward[i] + self.gamma * np.amax(self.target_net.predict(next_state[i],batch_size=len(minibatch))[0]))


            q_table = self.target_net.predict(np.squeeze(state, axis=1), batch_size=len(minibatch))

            #test = q_table[:,np.asarray(action)]
            #test2 = np.array(q_value)
            #q_table[:][np.asarray(action)] = q_value[:]
            for i in range(0, len(minibatch)):
                q_table[i][action[i]] = q_value[i]

            #vill ta

            history = self.model.fit(np.squeeze(state, axis=1), q_table, epochs=1, verbose=0)

            previousdata.append(history.history['loss'][-1])
            lossmean = sum(previousdata)/previousdata.__len__()
            print lossmean


            if minloss>=lossmean and len(previousdata)==previousdata.maxlen:
                print "training  completed"
                self.target_net.set_weights(self.model.get_weights()) # Copy weights to target net
                break



if __name__ == "__main__":
    env = gym.make('Breakout-v4')
    env = wr.FireResetEnv(env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    DQN = DQN(state_size,action_size)
    episodes = 10000000

    pre = PreProcessing()
    rew_max = 0
    avg_100rew=[]
    states_in_mem = 0 
    for i in range(episodes):



        new_episode = True
        state = env.reset()
        state = pre.proc(state, new_episode)
        new_episode=False
        done = False
        time = 0
        tot_rew=0
        if DQN.epsilon > DQN.epsilon_min:
            DQN.epsilon *= DQN.epsilon_decay

        while not done:

            #env.render()
            action = DQN._predict(state)
            #print DQN.model.predict(state)

            next_state, reward, done, lives = env.step(action)
            if lives['ale.lives'] != 5:
                done = True;
            #experience replay
            tot_rew += reward
            #last_state = currystate
            next_state = pre.proc(next_state,new_episode)
            
            if not done:
                DQN._append_mem(state, action, reward, next_state, done)
            else:
                next_state = np.zeros((1,80,80,2),dtype=np.uint8)
                if tot_rew > rew_max:
                    rew_max = tot_rew
                DQN._append_mem(state, action, reward, next_state, done)

                avg_100rew.append(tot_rew)
                if i%100 ==0:
                    avg100 = sum(avg_100rew)/len(avg_100rew)
                print("episode: {}/{}, reward: {}, epsilon: {:.2}, max reward: {}, mean past 100 rewards: {:.2}\n".format(i, episodes, tot_rew, DQN.epsilon,rew_max, avg100))
                if i%100 ==0:
                    del avg_100rew[:]
            time += 1
            states_in_mem += 1
            state = next_state
            if states_in_mem >= 100000:
                if states_in_mem >= DQN.batch_size:
                    DQN._train()
                    DQN.model.save_weights('my_weights.model')
                    print("Saving weights")
                    DQN.epsilon = 1.0
                    states_in_mem=0



