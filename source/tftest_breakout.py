import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D
from keras.optimizers import Adam
from keras import optimizers
from keras import initializers
from keras.models import Model

import matplotlib.pyplot as plt 
from collections import deque
from keras.utils import plot_model
import random
from PIL import Image
import torchvision.transforms as transf
import wrappers as wr

import keras.backend as K

import tensorflow as tf
#np.random.seed(1234)


height=105
width=80
toPlot = False
env = gym.make('SpaceInvaders-v4')
transform = transf.Compose([transf.ToPILImage(), transf.Resize((height,width)), transf.Grayscale(1)])

possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

class PreProcessing:
    def __init__(self):
        self.past_frames = deque([np.zeros((height,width), dtype=np.uint8) for i in xrange(1)], maxlen=4)
        #self.transform = transf.Compose([transf.ToPILImage(), transf.Resize((height,width)), transf.Grayscale(1)])
        self.past_img_buffer = np.zeros((1,105,80))

    def rescale_crop(self,frame):
         img = transform(frame)
         img = np.array(img)
         #proc_img = img[17:97,:]
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
    #state = np.array(proc_state).reshape((1, height, width))

    if new_episode:
        stacked_frames = deque([np.zeros((105, 80), dtype=np.uint8) for i in xrange(4)], maxlen=4)
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)

        stacked_state = np.stack(stacked_frames, axis=-1)
    else:
        stacked_frames.append(state)
        stacked_state = np.stack(stacked_frames, axis=-1)

    return stacked_frames, stacked_state #1*80*80*4

class DQNetwork:
    def __init__(self, state_size, action_size, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.input_dim = [height,width,4]
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        self.gamma = 0.97
        self.memory = deque(maxlen=200000)
        self.learning_rate = 0.001
        self.batch_size = 64
        self.states_in_mem = 0
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, 105, 80, 4], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            
            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out, filters = 64, kernel_size = [4,4], strides = [2,2], padding = "VALID", kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name = "conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")            
            
            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,filters = 64,kernel_size = [3,3],
strides = [2,2],  padding = "VALID",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),name = "conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            
            self.fc = tf.layers.dense(inputs = self.flatten, units = 512, activation = tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")      
            self.output = tf.layers.dense(inputs = self.fc, kernel_initializer=tf.contrib.layers.xavier_initializer(), units = self.action_size, activation=None)
            
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
            
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _append_mem(self, state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state, done))
        self.states_in_mem += 1 

    def _predict(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = sess.run(DQN.output, feed_dict = {DQN.inputs_: state.reshape((1,height,width,4))})
        action = np.argmax(q_values)
        #print act_values
        return action

    def _train(self):
        minibatch = random.sample(self.memory, self.batch_size)
        state =  np.array([each[0] for each in minibatch], ndmin=3)
        action = np.array([each[1] for each in minibatch])
        reward = np.array([each[2] for each in minibatch])
        next_state = np.array([each[3] for each in minibatch], ndmin=3)
        done = np.array([each[4] for each in minibatch])
        q_value =[]
        q_next_state = sess.run(DQN.output, feed_dict = {DQN.inputs_: next_state})

        for j in xrange(0, len(minibatch)):
            if done[j] == True:
                q_value.append(reward[j])
            else:
                target = reward[j] + self.gamma * np.max(q_next_state[j])
                q_value.append(target)

        targets = np.array([each for each in q_value])

        loss, _ = sess.run([DQN.loss, DQN.optimizer], feed_dict={DQN.inputs_: state, DQN.target_Q: targets, DQN.actions_: action})


if __name__ == "__main__":
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    
    tf.reset_default_graph()

    #DQN = DQN(state_size,action_size)
    DQN = DQNetwork(state_size, action_size)
    
    saver = tf.train.Saver()
    
    episodes = 1000000
    stacked_frames = deque([np.zeros((105, 80), dtype=np.uint8) for i in xrange(4)], maxlen=4)
    #inst_mem(DQN,stacked_frames)

    for i in xrange(DQN.batch_size):
        if i==0:
            state = env.reset()
            stacked_frames, state = preproc_stack(state, stacked_frames, True)
        action = env.action_space.sample()
        nextstate,rew,done,_=env.step(action)
        stacked_frames, nextstate = preproc_stack(nextstate, stacked_frames, False)
        action = possible_actions[action]
        if done:
            next_state = np.zeros((height,width,4),dtype=np.uint8)
            DQN._append_mem(state, action, rew, nextstate,done)
            state = env.reset()
            stacked_frames, state = preproc_stack(state, stacked_frames, True)
        else:
            DQN._append_mem(state, action, rew, nextstate,done)
            state = nextstate
            
    rew_max = 0
    avg_100rew=[]
    lives_left=5
    training = False
    if training == True:
        with tf.Session() as sess:
            # Initialize the variables
            sess.run(tf.global_variables_initializer())
            #saver.restore(sess, "./saved_weights.ckpt")
            for i in xrange(episodes):

                new_episode = True
                state = env.reset()
                stacked_frames, state = preproc_stack(state, stacked_frames,new_episode)    
                new_episode=False
                done = False
                time = 0
                tot_rew=0
                if DQN.epsilon > DQN.epsilon_min:
                    DQN.epsilon *= DQN.epsilon_decay
    
                while not done:
                    #env.render()
                    action = DQN._predict(state)
    
                    next_state, reward, done, lives = env.step(action)
                    tot_rew += reward

                    # action as hot encoded array
                    action = possible_actions[action]

                    
                    if not done:
                        stacked_frames, next_state = preproc_stack(next_state, stacked_frames,new_episode)
                        DQN._append_mem(state, action, reward,next_state, done)
                        state = next_state
                    else:
                        next_state = np.zeros((height,width,4),dtype=np.uint8)
                        #stacked_frames, next_state = preproc_stack(next_state, stacked_frames,new_episode)
                        DQN._append_mem(state, action, reward,next_state, done)

                        if tot_rew > rew_max:
                            rew_max = tot_rew
                        
                        avg_100rew.append(tot_rew)
                        avg100 = sum(avg_100rew)/len(avg_100rew)
                        print("episode: {}/{}, transitions: {}/200000 reward: {}, epsilon: {:.2}, max reward: {}, mean past 100 rewards: {:.4f}\n".format(i, episodes, DQN.states_in_mem, tot_rew, DQN.epsilon,rew_max, avg100))
                        if i%100 ==0:
                            del avg_100rew[:]
                        
                    DQN._train()
                if i % 5 == 0:
                    save_path = saver.save(sess, "./saved_weights3.ckpt")
                    print("Model Saved")


    with tf.Session() as sess:
        total_test_rewards = []
    
        # Load the model
        saver.restore(sess, "./saved_weights3.ckpt")
        #sess.run(tf.global_variables_initializer())
        DQN.epsilon = 0.01
        for episode in xrange(20):
            total_rewards = 0
        
            state = env.reset()
            stacked_frames, state = preproc_stack(state, stacked_frames,True)
        
            print("****************************************************")
            print("EPISODE ", episode)
        
            while True:
                # Reshape the state
                state = state.reshape((1, 105, 80, 4))

                action = DQN._predict(state)
            
            #Perform the action and get the next_state, reward, and done information
                next_state, reward, done, _ = env.step(action)
                env.render()
            
                total_rewards += reward

                if done:
                    print ("Score", total_rewards)
                    total_test_rewards.append(total_rewards)
                    break
                
                
                stacked_frames, next_state = preproc_stack(next_state, stacked_frames,False)
                state = next_state
        avg = sum(total_test_rewards)/len(total_test_rewards)
        print("Average score", avg)        
        env.close()

