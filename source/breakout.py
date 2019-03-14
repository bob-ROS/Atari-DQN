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
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        self.gamma = 0.97
        self.memory = deque(maxlen=200000)
        self.learning_rate = 0.001
        self.batch_size = 128
        self.model = self._create_model()
        self.states_in_mem = 0
        self.q_values = np.zeros(shape=[200000, act_size], dtype=np.float)
        self.estimation_errors = np.zeros(shape=200000, dtype=np.float)


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
        q_values = self.model.predict(state)
        action = np.argmax(q_values)
        #print act_values
        self.q_values[self.states_in_mem] = q_values
        return action

    def _train(self):
        self.update_all_q_values()
        minloss = 0.1;
        #previousdata= deque(maxlen=100)
        #Do something with mean
        min_epochs=1.0
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
            state =  np.array([each[0] for each in minibatch], ndmin=4)
            action = np.array([each[1] for each in minibatch])
            reward = np.array([each[2] for each in minibatch])
            next_state = np.array([each[3] for each in minibatch], ndmin=4)
            done = np.array([each[4] for each in minibatch])
            q_value =[]



            for j in range(0, len(minibatch)):
                if done[j] == True:
                    q_value.append(reward[j])
                else:
                  q_next_state = self.model.predict(next_state[j], batch_size=len(minibatch))[0]
                  q_value.append(reward[j] + self.gamma * np.amax(q_next_state))

            state = np.squeeze(state, axis=1)
            q_table = self.model.predict(state, batch_size=len(minibatch))

            for k in range(0, len(minibatch)):
                q_table[k][action[k]] = q_value[k]

            history = self.model.fit(state, q_table, epochs=1, verbose=0)

            loss_history = np.roll(loss_history, 1)
            loss_history[0] = history.history['loss'][-1]

            # Calculate the average loss for the previous batches.
            loss_mean = np.mean(loss_history)

            # Print status.
            pct_epoch = i / iterations_per_epoch
            print("\tIteration: {0}/min_iter: {1} ({2:.2f} epoch), Batch loss: {3:.4f}, Mean loss: {4:.4f}".format(i,min_iterations, pct_epoch, history.history['loss'][-1], loss_mean))
            if i > min_iterations and loss_mean < minloss:
                print "training  completed"
                break

    def update_all_q_values(self):
        """
        Update all Q-values in the replay-memory.
        
        When states and Q-values are added to the replay-memory, the
        Q-values have been estimated by the Neural Network. But we now
        have more data available that we can use to improve the estimated
        Q-values, because we now know which actions were taken and the
        observed rewards. We sweep backwards through the entire replay-memory
        to use the observed data to improve the estimated Q-values.
        """

        # Copy old Q-values so we can print their statistics later.
        # Note that the contents of the arrays are copied.
        #self.q_values_old[:] = self.q_values[:]

        # Process the replay-memory backwards and update the Q-values.
        # This loop could be implemented entirely in NumPy for higher speed,
        # but it is probably only a small fraction of the overall time usage,
        # and it is much easier to understand when implemented like this.
        #(state, action, reward, next_state, done,qval)
        for k in range(len(self.memory)-1):
            # Get the data for the k'th state in the replay-memory.
            action = self.memory[k][1]
            reward = self.memory[k][2]
            end_life = self.memory[k][4]
            #end_episode = self.end_episode[k]

            # Calculate the Q-value for the action that was taken in this state.
            if done:
                # If the agent lost a life or it was game over / end of episode,
                # then the value of taking the given action is just the reward
                # that was observed in this single step. This is because the
                # Q-value is defined as the discounted value of all future game
                # steps in a single life of the agent. When the life has ended,
                # there will be no future steps.
                action_value = reward
            else:
                # Otherwise the value of taking the action is the reward that
                # we have observed plus the discounted value of future rewards
                # from continuing the game. We use the estimated Q-values for
                # the following state and take the maximum, because we will
                # generally take the action that has the highest Q-value.
                action_value = reward + self.gamma * np.max(self.q_values[k + 1])

            # Error of the Q-value that was estimated using the Neural Network.
            self.estimation_errors[k] = abs(action_value - self.q_values[k, action])

            # Update the Q-value with the better estimate.
            self.q_values[k, action] = action_value

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
    lives_left=5
    
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

            next_state, reward, done, lives = env.step(action)
            tot_rew += reward

            next_state = pre.proc(next_state,new_episode)
            
            if not done:
                DQN._append_mem(state, action, reward, next_state, done)
            else:
                next_state = np.zeros((1,105,80,2),dtype=np.uint8)
                if tot_rew > rew_max:
                    rew_max = tot_rew
                DQN._append_mem(state, action, reward, next_state, done)
                avg_100rew.append(tot_rew)
                #if i%100 ==0:
                avg100 = sum(avg_100rew)/len(avg_100rew)
                print("episode: {}/{}, transitions: {}/200000 reward: {}, epsilon: {:.2}, max reward: {}, mean past 100 rewards: {:.2}\n".format(i, episodes, DQN.states_in_mem, tot_rew, DQN.epsilon,rew_max, avg100))
                if i%100 ==0:
                    del avg_100rew[:]
            time += 1
            state = next_state
            if DQN.states_in_mem >= 19900:
                if DQN.states_in_mem >= DQN.batch_size:
                    DQN._train()
                    DQN.model.save_weights('my_weights.model')
                    print("Saving weights")
                    DQN.epsilon = 1.0
                    DQN.states_in_mem=0



