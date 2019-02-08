import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt 
from collections import deque
from keras.utils import plot_model
import random


#unprocessed_image_size = fromgame

#preprocessed_image_size = x

np.random.seed(1234)

# return observationProcessed(imagefromstep)
def observationProcessing(env, new_episode = False):
    screen = env.render(mode='rgb_array')
    #screen = np.mean(screen,-1)
    screen = screen.mean(-1)
    #screen = screen[0:350][150:450]
    screen = screen[150:350,50:550]
    plt.imshow(screen, cmap = plt.get_cmap('gray'))
    plt.show()
    #print screen.shape
    #screen = np.expand_dims(screen, axis=0)
    #screen = np.expand_dims(screen, axis=2)
    screen = screen.reshape(( 200, 500, 1))

    #plt.imshow(screen[:,:,0])
    #plt.show()
    if new_episode:
        stacked_frames = deque([np.zeros((200, 50), dtype=np.int) for i in range(4)], maxlen=4)
        s_t = np.stack((screen, screen, screen, screen), axis=2)

    else:
        




    #print (s_t.shape)

    #In Keras, need to reshape



    return s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*200*600*4




class DQN:
    def __init__(self, state_size,act_size):
        self.state_size = state_size
        self.act_size = act_size
        self.input_dim = [0,200,600,1]
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01
        self.gamma = 0.90
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.batch_size = 16
        self.model = self._create_model()


    def _create_model(self):
        model = Sequential()
        #model.add(Convolution2D(input_shape = (200, 600, 1), filters=32,kernel_size=8,strides=(4,4),padding='valid', data_format='channels_last', kernel_initializer='zeros',bias_initializer='zeros',activation='relu'))
        #model.add(Convolution2D(filters=64,kernel_size=4,strides=(2,2),padding='valid', data_format='channels_last',kernel_initializer='zeros', bias_initializer='zeros', activation='relu'))
        #model.add(Convolution2D(filters=64, kernel_size=4, strides=(1, 1), padding='valid', kernel_initializer='zeros',bias_initializer='zeros', data_format='channels_last', activation='relu'))
        #model.add(Flatten())
        #model.add(Dense(units=512,activation='relu',kernel_initializer='zeros', bias_initializer='zeros'))
        #model.add(Dense(units=self.act_size,activation='linear',kernel_initializer='he_uniform', bias_initializer='zeros'))
        model.add(Convolution2D(32, (8, 8), subsample=(4, 4),  activation='relu',input_shape=(200, 500, 4)))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2) ,activation='relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1) ,activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.act_size,  activation='relu'))
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
            q_value = -5 # assume punishment
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
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    DQN = DQN(state_size,action_size)
    plot_model(DQN.model, to_file='model.png',  show_layer_names=True, show_shapes=True)
    episodes = 1000000
    for i in range(episodes):
        env.reset()
        done = False
        #currystate = observationProcessing(env)
        #pastastate = observationProcessing(env)
        #state = currystate - pastastate
        #plt.imshow(state)
        #plt.show()
        time = 0
        state = observationProcessing(env, new_episode=True)
        #plt.imshow(state[0,:,:,0], cmap='gray')
        #plt.show()
        while not done:
            #state = observationProcessing(env)
            action = DQN._predict(state)
            #print DQN.model.predict(state)

            _, reward, done, _ = env.step(action)
            #experience replay

            #last_state = currystate
            next_state = observationProcessing(env)
            if not done:
                #next_state = currystate - last_state
                #next_state = currystate
                DQN._append_mem(state, action, reward, next_state, done)
            else:
                #print "reward: {}".format(reward)
                #reward = -reward #punishment sufficient?
                next_state = 0
                DQN._append_mem(state, action, reward, next_state, done)
                if len(DQN.memory) >= DQN.batch_size:
                    # print "eat shit python cucks"
                    DQN._train()
                print("episode: {}/{}, score: {}, epsilon: {:.2}".format(i, episodes, time, DQN.epsilon))

            time += 1
            state = next_state


