import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D
from keras.optimizers import Adam
from keras import optimizers
from keras import initializers
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
        self.past_frames = deque([np.zeros((105, 80), dtype=np.uint8) for i in range(1)], maxlen=4)
        self.transform = transf.Compose([transf.ToPILImage(), transf.Resize((105,80)), transf.Grayscale(1)])

    def rescale_crop(self,frame):
         img = self.transform(frame)
         img = np.array(img)
         #proc_img = img[17:97,:]
         proc_img = np.array(img).reshape((1, 105, 80))
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
        self.epsilon = 0.01
        self.epsilon_decay = 0.0
        self.epsilon_min = 0.01
        self.gamma = 0.90
        self.memory = deque(maxlen=200000)
        self.learning_rate = 0.001
        self.batch_size = 128
        self.model = self._create_model()
        self.states_in_mem = 0


    def _create_model(self):

        init = initializers.TruncatedNormal(mean=0.0, stddev=2e-2, seed=None)
        model = Sequential()
        model.add(Convolution2D(16, 3, strides=2,padding='same', kernel_initializer=init,activation='relu',input_shape=(105, 80, 2)))
        model.add(Convolution2D(32, 3, strides=2,padding='same' ,activation='relu',kernel_initializer=init))
        model.add(Convolution2D(64, 3, strides=1,padding='same' ,activation='relu',kernel_initializer=init))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu',kernel_initializer=init))
        model.add(Dense(1024, activation='relu',kernel_initializer=init))
        model.add(Dense(1024, activation='relu',kernel_initializer=init))
        model.add(Dense(1024, activation='relu',kernel_initializer=init))
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
        self.states_in_mem += 1 

    def _predict(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.act_size)
        act_values = self.model.predict(state)
        action = np.argmax(act_values)
        #print act_values
        return action

    def _train(self):
        minloss = 0.15;
        #previousdata= deque(maxlen=100)
        #Do something with mean
        min_epochs=1
        max_epochs=10

        # Number of optimization iterations corresponding to one epoch.
        iterations_per_epoch = self.states_in_mem / self.batch_size

        # Minimum number of iterations to perform.
        min_iterations = int(iterations_per_epoch * min_epochs)

        # Maximum number of iterations to perform.
        max_iterations = int(iterations_per_epoch * max_epochs)

        # Buffer for storing the loss-values of the most recent batches.
        loss_history = np.zeros(100, dtype=float)

        for i in range(max_iterations):
            minibatch = random.sample(self.memory, self.batch_size)
            #minibatch = self.memory
            for state, action, reward, next_state, done in minibatch:
                q_value = reward # assume punishment
                if not done:
                    q_value = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

                q_table = self.model.predict(state)

                q_table[0][action] = q_value

                history = self.model.fit(state, q_table, epochs=1, verbose=0)

                loss_history = np.roll(loss_history, 1)
                loss_history[0] = history.history['loss'][-1]

                # Calculate the average loss for the previous batches.
                loss_mean = np.mean(loss_history)

                # Print status.
                pct_epoch = i / iterations_per_epoch
                print("\tIteration: {0} ({1:.2f} epoch), Batch loss: {2:.4f}, Mean loss: {3:.4f}".format(i, pct_epoch, history.history['loss'][-1], loss_mean))

                # Stop the optimization if we have performed the required number
                # of iterations and the loss-value is sufficiently low.
                if i > min_iterations and loss_mean < minloss:
                    break
"""
                history = self.model.fit(state, q_table, epochs=1, verbose=0)
                previousdata.append(history.history['loss'][-1])
                lossmean = sum(previousdata)/previousdata.__len__()
                print("min loss: {:.2}, loss mean: {:.2}, prev data len: {}/100".format(minloss, lossmean, previousdata.__len__()))


            if minloss >= lossmean and previousdata.__len__() == previousdata.maxlen:
                print "training  completed"
                break
"""



if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v4')
    #env = wr.FireResetEnv(env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    DQN = DQN(state_size,action_size)
    episodes = 10000000

    pre = PreProcessing()
    rew_max = 0
    avg_100rew=[]
    states_in_mem = 0 
    lives_left=5
    for i in range(episodes):

        new_episode = True
        state = env.reset()
        state = pre.proc(state, new_episode)
        new_episode=False
        done = False
        tot_rew=0

        while not done:

            env.render()
            action = DQN._predict(state)
            #print DQN.model.predict(state)

            next_state, reward, done, lives = env.step(action)
            if lives['ale.lives'] < lives_left:
                lives_left = lives['ale.lives']
                next_state, reward, done, lives = env.step(0)
            #experience replay
            tot_rew += reward
            #last_state = currystate
            next_state = pre.proc(next_state,new_episode)
            
            if done:
                next_state = 0
                if tot_rew > rew_max:
                    rew_max = tot_rew
                avg_100rew.append(tot_rew)
                if i%100 ==0:
                    avg100 = sum(avg_100rew)/len(avg_100rew)
                print("episode: {}/{}, reward: {}, epsilon: {:.2}, max reward: {}, mean past 100 rewards: {:.1}\n".format(i, episodes, tot_rew, DQN.epsilon,rew_max, avg100))
                if i%100 ==0:
                    del avg_100rew[:]
            state = next_state



