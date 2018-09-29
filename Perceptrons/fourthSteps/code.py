import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

Y_train = np_utils.to_categorical( y_train, 10 )
Y_test = np_utils.to_categorical( y_test, 10 )

X_train = x_train.reshape([-1,28*28]) / 255.
X_test = x_test.reshape([-1,28*28]) / 255.


def createModel(init):
	model = Sequential()
	model.add( Dense(100, input_shape=(28*28,), init=init, activation='tanh'))
	model.add( Dense(100, init=init, activation='tanh'))
	model.add( Dense(100, init=init, activation='tanh'))
	model.add( Dense(100, init=init, activation='tanh'))
	model.add( Dense(10, init=init, activation='softmax'))
	return model

uniform_model = createModel("uniform")
uniform_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
uniform_model.fit(X_train, Y_train, batch_size=64, nb_epoch=30, verbose=1, validation_data=(X_test,Y_test))

uniform_model = createModel("glorot_normal")
uniform_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
uniform_model.fit(X_train, Y_train, batch_size=64, nb_epoch=30, verbose=1, validation_data=(X_test,Y_test))