import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt 


#unprocessed_image_size = fromgame

#preprocessed_image_size = x


# return observationProcessed(imagefromstep)
def observationProcessing(env):
    env.reset()
    screen = env.render(mode='rgb_array')
    #screen = np.mean(screen,-1)
    screen = screen.mean(-1)
    plt.imshow(screen, cmap = plt.get_cmap('gray'))
    plt.show()
#initialize convolutional network


#save / load network


#experience_replay


# MAIN

#loop
	#add.parser visualize or not

	#blah = gym.step()
	#obsevationProcessed(blah)

	#experience_replay

env = gym.make('CartPole-v0')
observationProcessing(env)

#if __name__ == "__main__":

