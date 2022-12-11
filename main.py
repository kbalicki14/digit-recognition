import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D

from keras.callbacks import EarlyStopping, ModelCheckpoint


(x_train, y_train), (x_test, y_test) = mnist.load_data()


def plot_input_img(i):
    plt.imshow(x_train[i], cmap='binary')
    plt.title(y_train[i])
    plt.show()


# for i in range(10):
#      plot_input_img(i)

#pre proces data

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# print(x_train)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


#build model

model = Sequential()

model.add(Conv2D(32, (3,3),input_shape=(28,28,1), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))

#print(model.summary())
model.compile(optimizer='adam', loss = keras.losses.categorical_crossentropy, metrics=['accuracy'])

es = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4, verbose=1)
mc = ModelCheckpoint("./model_mnist.h5", "val_acc", verbose=1, save_best_only=True)

cb = [es,mc]

# # trainig model

his = model.fit(x_train, y_train, epochs=5, validation_split=0.3)

model.save('model_mnist.h5')
print("Saving the model as mnist.h5")

model_load = keras.models.load_model('model_mnist.h5')
score = model_load.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])










