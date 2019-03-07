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
def observationProcessing(env):
    screen = env.render(mode='rgb_array')
    #screen = np.mean(screen,-1)
    screen = screen.mean(-1)
    #screen = screen[0:350][150:450]
    screen = screen[150:350][:]
    #plt.imshow(screen, cmap = plt.get_cmap('gray'))
    #plt.show()
    #print screen.shape
    #screen = np.expand_dims(screen, axis=0)
    #screen = np.expand_dims(screen, axis=2)
    screen = screen.reshape((-1, 200, 600, 1))
    return screen

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        #self.input_dim = [0,200,600,1]
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01
        self.gamma = 0.90
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.0001
        self.batch_size = 16
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

    def _append_mem(self, state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state, done))

    def _predict(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        action = np.argmax(act_values)
   
        return action

    def _train(self):
        minibatch = random.sample(self.memory, self.batch_size)
        #minibatch = self.memory
        for state, action, reward, next_state, done in minibatch:
            q_value = -5 # assume punishment
            if not done: #If survived
                q_value = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            q_table = self.model.predict(state)  
            q_table[0][action] = q_value
            self.model.fit(state, q_table, epochs=10, verbose=0) #Epochs = 10 iterations trained, not necessasary
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
   
        done = False
  
        time = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
          
            action = DQN._predict(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            next_state = np.reshape(next_state, [1, state_size])
        

            if not done:
 
                DQN._append_mem(state, action, reward, next_state, done)
            else:
  
                next_state = 0
                DQN._append_mem(state, action, reward, next_state, done)
                if len(DQN.memory) >= DQN.batch_size:
                    # print "eat shit python cucks"
                    DQN._train()
                print("episode: {}/{}, score: {}, epsilon: {:.2}".format(i, episodes, time, DQN.epsilon))

            time += 1
            state = next_state



