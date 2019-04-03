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
import wrappers as wr
from keras.models import Model
import keras.backend as K

import tensorflow as tf
#np.random.seed(1234)
height=105
width=80
toPlot = False
toRender = True
env = gym.make('SpaceInvaders-v4')
transform = transf.Compose([transf.ToPILImage(), transf.Resize((height,width)), transf.Grayscale(1)])

possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

class PreProcessing:
    def __init__(self):
        self.past_frames = deque([np.zeros((height,width), dtype=np.uint8) for i in range(1)], maxlen=4)
        self.past_img_buffer = np.zeros((1,105,80))

    def rescale_crop(self,frame):
         img = transform(frame)
         img = np.array(img)

         proc_img = np.array(img).reshape((1, height, width))
         return proc_img

    def proc(self,frame,new_ep):
        proc_image = self.rescale_crop(frame)
        self.past_frames.append(proc_image)
        if new_ep:
            diff = proc_image
        else:
            diff = proc_image - self.past_frames[0]
            self.past_img_buffer = diff
        motion = diff + 0.5*self.past_img_buffer    
        state = np.stack([proc_image, motion], axis=-1)
        return state

def preproc_stack(state, stacked_frames,new_episode = False ):
    proc_state = transform(state)
    state = np.array(proc_state)
    #state = proc_state[17:97,:]
    #state = np.array(proc_state).reshape((height, width))

    if new_episode:
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_state = np.stack(stacked_frames, axis=-1)
    else:
        stacked_frames.append(state)
        stacked_state = np.stack(stacked_frames, axis=-1)

    #In Keras, need to reshape

    return stacked_frames, stacked_state #1*80*80*4


class DQN:
    def __init__(self, state_size,act_size):
        self.state_size = state_size
        self.act_size = act_size
        self.input_dim = [height,width,4]
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        self.gamma = 0.97
        self.memory = deque(maxlen=200000)
        self.learning_rate = 0.001
        self.batch_size = 128
        self.numberofnets = act_size
        self.model = []
        self.model.append(self._create_model())  #assumes actions > 0, else this will be weird asf
        for i in xrange(1,act_size): #
            self.model.append(self._create_model())
            self.model[i].set_weights(self.model[0].get_weights())



        self.states_in_mem = 0
        self.q_values = np.zeros(shape=[200000, act_size], dtype=np.float)
        self.estimation_errors = np.zeros(shape=200000, dtype=np.float)



    def _create_model(self):

        init = initializers.TruncatedNormal(mean=0.0, stddev=2e-2, seed=None)
        model = Sequential()
        model.add(Convolution2D(16, 3, strides=2,padding='same', kernel_initializer=init, name='1conv16', activation='relu',input_shape=self.input_dim))
        model.add(Convolution2D(32, 3, strides=2,padding='same' ,activation='relu',kernel_initializer=init, name='2conv32'))
        model.add(Convolution2D(64, 3, strides=1,padding='same' ,activation='relu',kernel_initializer=init, name='3conv64'))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu',kernel_initializer=init))
        model.add(Dense(1024, activation='relu',kernel_initializer=init))
        model.add(Dense(1024, activation='relu',kernel_initializer=init))
        model.add(Dense(1024, activation='relu',kernel_initializer=init))
        model.add(Dense(1,  activation='linear'))
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
        Qvalues = []
        for i in xrange(self.act_size):
            Qvalues.append(self.model[i].predict(np.expand_dims(state,axis=0)))

        action = np.argmax(Qvalues)
        #action = np.argmax(q_values)

        #self.q_values[self.states_in_mem] = q_values
        return action

    def _train(self):
            minibatch = random.sample(self.memory, self.batch_size)
            state =  np.array([each[0] for each in minibatch], ndmin=3)
            action = np.array([each[1] for each in minibatch])
            reward = np.array([each[2] for each in minibatch])
            next_state = np.array([each[3] for each in minibatch], ndmin=3)
            done = np.array([each[4] for each in minibatch])
            q_value =[]

            for j in xrange(len(minibatch)):
                if done[j] == True:
                    q_value.append(reward[j])
                else:
                    q_next_state = []
                    for i in xrange(self.act_size):
                        temp = self.model[action[i]].predict(np.expand_dims(next_state[j],axis=0))
                        q_next_state.append(temp)
                    q_value.append(reward[j] + self.gamma * np.amax(q_next_state))

            for i in xrange(len(minibatch)):
                input_image = np.expand_dims(state[i],axis=0)
                input_reward = np.array(q_value[i])
                input_reward = input_reward[np.newaxis]

                history = self.model[action[i]].fit(input_image, input_reward, epochs=1, verbose=0)
                print("\tIteration: {} Mean loss: {:.2f}".format(i,history.history['loss'][0],))
            #print("\tIteration: {0}/min_iter: {1} ({2:.2f} epoch), Batch loss: {3:.4f}, Mean loss: {4:.4f}".format(i,min_iterations, pct_epoch, history.history['loss'][-1], loss_mean))
            #if i > min_iterations and loss_mean < minloss:
            #    print "training  completed"
            #    break

def plot_conv_weights(model, layer):
    W = model.get_layer(name=layer).get_weights()[0]
    if len(W.shape) == 4:
        W = np.squeeze(W)
        W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3])) 
        fig, axs = plt.subplots(5,5, figsize=(8,8))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for i in range(25):
            axs[i].imshow(W[:,:,i])
            axs[i].set_title(str(i))
        plt.show()
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    print("Dimensions of images: {}".format(activation.shape))
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(10,10))#(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
    plt.show()

if __name__ == "__main__":


    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    DQN = DQN(state_size,action_size)
    episodes = 1000000
    stacked_frames = deque([np.zeros((105, 80), dtype=np.int) for i in range(4)], maxlen=4)
    pre = PreProcessing()
    rew_max = 0
    avg_100rew=[]
    lives_left=5

    if toPlot == True:
        """PLOTTING OUTPUTS START"""
        #state = pre.proc(env.reset(), True)
        stacked_frames, state = preproc_stack(env.reset(), stacked_frames,True)
        for k in xrange(1000):
            action = DQN._predict(state)
            nxt,_,dne,_ = env.step(action)
            #nxt=pre.proc(nxt,False)
            stacked_frames, nxt = preproc_stack(nxt, stacked_frames,False)
            prv_state = state
            state = nxt
            if dne:
                break

        #plot_conv_weights(DQN.model,'2conv32')
        layer_outputs = [layer.output for layer in DQN.model.layers]
        activation_model = Model(inputs=DQN.model.input, outputs=layer_outputs)
        activations = activation_model.predict(state,True)


        state = np.squeeze(state, axis=0)
        fig,ax = plt.subplots(1,4, figsize=(20,20))
        ax[0].imshow(state[:,:,0], cmap='gray')
        ax[1].imshow(state[:,:,1], cmap='gray')
        ax[2].imshow(state[:,:,2], cmap='gray')
        ax[3].imshow(state[:,:,3], cmap='gray')
        plt.show()
        display_activation(activations, 4, 4, 0)
        display_activation(activations, 4, 8, 1)
        display_activation(activations, 8, 8, 2)
        """PLOTTING OUTPUTS END"""

    for i in xrange(episodes):

        new_episode = True
        state = env.reset()
        #state = pre.proc(state, new_episode)
        stacked_frames, state = preproc_stack(state, stacked_frames,new_episode)    
        new_episode=False
        done = False
        time = 0
        tot_rew=0
        if DQN.epsilon > DQN.epsilon_min:
            DQN.epsilon *= DQN.epsilon_decay

        while not done:
            if toRender == True:
                env.render()
            action = DQN._predict(state)

            next_state, reward, done, lives = env.step(action)
            tot_rew += reward

            #next_state = pre.proc(next_state,new_episode)
            stacked_frames, next_state = preproc_stack(next_state, stacked_frames,new_episode)
            
            if not done:
                DQN._append_mem(state, action, reward, next_state, done)
            else:
                next_state = np.zeros((1,height,width,4),dtype=np.uint8)
                if tot_rew > rew_max:
                    rew_max = tot_rew
                DQN._append_mem(state, action, reward, next_state, done)
                avg_100rew.append(tot_rew)
                #if i%100 ==0:
                avg100 = sum(avg_100rew)/len(avg_100rew)
                print("episode: {}/{}, transitions: {}/200000 reward: {}, epsilon: {:.2}, max reward: {}, mean past 100 rewards: {:.4f}\n".format(i, episodes, DQN.states_in_mem, tot_rew, DQN.epsilon,rew_max, avg100))
                if i%100 ==0:
                    del avg_100rew[:]
            time += 1
            state = next_state
            if DQN.states_in_mem >= 100000:
                if DQN.states_in_mem >= DQN.batch_size:
                    DQN._train()
                    #DQN.model.save_weights('my_weights.model') # FIX GOOD WAY OF SAVING!
                    print("Saving weights")
                    DQN.epsilon = 1.0
                    DQN.states_in_mem=0



