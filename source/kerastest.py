from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
import gym

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
env = gym.make('BeamRider-v0')
env.render()

plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)





raw_input("Press Enter to continue")