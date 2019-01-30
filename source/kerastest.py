from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
import gym
import time

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
env = gym.make('BeamRider-v0')


plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)


env.reset()
for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random acti
    print("Reward: %.10f" %(reward))
    time.sleep(0.1)



raw_input("Press E to continue")
