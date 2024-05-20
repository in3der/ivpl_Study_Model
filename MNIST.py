from __future__ import print_function
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras import backend as K
# from keras.utils import np_utils
from PIL import Image
import numpy as np
import os
from tensorflow.keras.callbacks import TensorBoard
tensorboard_callback = TensorBoard(log_dir='C:/Users/jyh22/PycharmProjects/myProject/logs', histogram_freq=1)


batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), strides=1, activation='relu', input_shape=input_shape, padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(16, kernel_size=(5, 5), strides=1, activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(120, kernel_size=(5, 5), strides=1, activation='relu'))
model.add(Flatten())
model.add(Dense(84, activation='tanh'))
model.add(Dense(num_classes, activation='softmax'))

def mse_loss(y_true, y_pred):
    err = y_true-y_pred
    loss = tf.math.reduce_mean(tf.math.square(err))
    return loss

model.summary()
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
model.compile(loss=mse_loss, optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=tensorboard_callback)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 그래프 그리기
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'val'])
plt.show()