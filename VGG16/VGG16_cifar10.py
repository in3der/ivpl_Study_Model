# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import tensorflow.python.keras as keras
import keras.metrics
# import keras
from absl import logging
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
import numpy as np
from tensorflow.keras import losses, layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D,
                                     MaxPool2D, Input, AveragePooling2D, Flatten,
                                     Dense, Activation, Lambda, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint, \
    LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import log_loss, top_k_accuracy_score
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# 텐서보드 사용
from datetime import datetime

root = '/home/ahlee/sichoi/VGG16'
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")

log_dir = f'/home/ahlee/sichoi/VGG16/logs/{timestamp}'
# tensorboard_callback = TensorBoard(log_dir=log_dir,
#                                   histogram_freq=1,
#                                   profile_batch=[500, 520])

logging.set_verbosity(logging.INFO)
AUTOTUNE = tf.data.experimental.AUTOTUNE

# GPU 장치 확인
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))

# 마지막 GPU를 사용하도록 설정
if physical_devices:
    try:
        # 마지막 GPU 선택
        last_gpu = physical_devices[-1]
        tf.config.experimental.set_visible_devices(last_gpu, 'GPU')
        print("Using GPU:", last_gpu)

        # 메모리 사용 동적 조정
        tf.config.experimental.set_memory_growth(last_gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available.")

# 모든 GPU 메모리 사용 허용
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class RandomColorAffine(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomColorAffine, self).__init__(**kwargs)

    def call(self, inputs):
        # RGB -> HSV
        hsv_image = tf.image.rgb_to_hsv(inputs)

        # 랜덤 색상 변화
        hue_delta = tf.random.uniform(shape=[], minval=-0.0005, maxval=0.0005)
        hsv_image = tf.image.adjust_hue(hsv_image, hue_delta)

        # HSV -> RGB
        output = tf.image.hsv_to_rgb(hsv_image)

        return output

    def apply_color_affine(self, inputs, hue_delta, saturation_factor, brightness_delta):
        hsv_image = tf.image.rgb_to_hsv(inputs)
        # hue delta만 변경
        hsv_image = tf.image.adjust_hue(hsv_image, hue_delta)
        hsv_image = tf.image.adjust_saturation(hsv_image, saturation_factor)
        hsv_image = tf.image.adjust_brightness(hsv_image, brightness_delta)
        outputs = tf.image.hsv_to_rgb(hsv_image)
        return outputs


data_preprocessing = tf.keras.Sequential([
    preprocessing.Rescaling(1. / 255., 1. / 255.),  # Rescale pixel values
    preprocessing.RandomCrop(224, 224),  # Random crop images to (224, 224)
    preprocessing.RandomFlip(mode='horizontal'),  # Randomly flip images horizontally
    # preprocessing.RandomContrast(factor=0.001),  # Randomly adjust contrast
    # RandomColorAffine(),
])

# 데이터 로드
batch_size = 32

# root2 = '/home/ivpl-d29/dataset/imagenet/'
# root2 = '/home/ivpl-d29/dataset/imagenet_mini/'


from keras.datasets import cifar10
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Split the data into training and validation sets using train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


def preprocess_images(x, y):
    x = tf.image.resize(x, (256, 256))
    return x, y


batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess_images).batch(batch_size).prefetch(
    AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(preprocess_images).batch(batch_size).prefetch(AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess_images).batch(batch_size).prefetch(
    AUTOTUNE)

data_preprocessing = tf.keras.Sequential([
    preprocessing.Rescaling(1. / 255.),  # 픽셀 값을 재조정
    preprocessing.RandomCrop(224, 224),  # 이미지를 (224, 224)로 무작위 자르기
    preprocessing.RandomFlip(mode='horizontal'),  # 이미지를 수평으로 무작위 뒤집기
    # preprocessing.RandomContrast(factor=0.001),  # 대비를 무작위로 조정
    RandomColorAffine(),  # RandomColorAffine 층 추가
])

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (data_preprocessing(x, training=False), y), num_parallel_calls=AUTOTUNE)


def subtract_mean(image):
    image = tf.cast(image, tf.float32)
    mean = tf.reduce_mean(image, axis=[1, 2], keepdims=True)
    centered_image = image - mean
    return centered_image


train_ds = train_ds.map(lambda x, y: (subtract_mean(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (subtract_mean(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (subtract_mean(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

l2 = tf.keras.regularizers.L2(l2=5e-4)
rd_normal = tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)
bias_0 = tf.keras.initializers.Zeros()

model = Sequential()
model.add(layers.InputLayer(input_shape=(224, 224, 3)))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(MaxPooling2D((2, 2), strides=(2, 2), ))
# Block 2
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(MaxPooling2D((2, 2), strides=(2, 2), ))
# Block 3
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(MaxPooling2D((2, 2), strides=(2, 2), ))
# Block 4)
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(MaxPooling2D((2, 2), strides=(2, 2), ))
# Block 5)
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal',
                 bias_initializer=bias_0, kernel_regularizer=l2))
model.add(MaxPooling2D((2, 2), strides=(2, 2), ))
# FC
model.add(Flatten())
model.add(Dense(4096, activation="relu", kernel_initializer='glorot_normal', ))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu", kernel_initializer='glorot_normal', ))
model.add(Dropout(0.5))
model.add(Dense(1000, activation="softmax", ))
model.build()
model.summary()

###################################################################
# top-K
tk_callback = tf.metrics.TopKCategoricalAccuracy(k=5)
sparse_tk_callback = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto', min_lr=0.0001,
                                       verbose=1)  # patience epoch이내에 줄어들지 않으면 factor만큼 감소
SGD = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, weight_decay=5e-4)
model.compile(optimizer=SGD, loss=losses.categorical_crossentropy, metrics=['accuracy', tk_callback])

# 체크포인트
checkpoint_path = root + "/ckpt/"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True,
                                                 verbose=1)

epochs = 20

'''
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=batch_size,
                    callbacks=[reduce_lr_callback, cp_callback, tensorboard_callback],
                    verbose=1)
score = model.evaluate(test_ds, verbose=1)
'''
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=batch_size,
                    callbacks=[reduce_lr_callback, cp_callback, ],
                    verbose=1)
score = model.evaluate(test_ds, verbose=1)

# 모델 세이브
model.save(os.path.join(checkpoint_path, 'model.h5'))
# 가중치 세이브
model.save_weights(os.path.join(checkpoint_path, 'weights.h5'))

print('Test loss :', score[0])
print('Test accuracy :', score[1])
print(score[0:])
print('Test top-1 error rate: ', 1 - score[1])
print('Test top-5 error rate: ', 1 - score[2])

# 그래프 그리기 - accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'val'])
plt.title('Accuracy')
# 그래프 그리기 - loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.title('Loss')
plt.show()

# 세션 Clear, 이전 메모리 반환
import keras.backend as K

K.clear_session()# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import tensorflow.python.keras as keras
import keras.metrics
#import keras
from absl import logging
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
import numpy as np
from tensorflow.keras import losses, layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D,
                          MaxPool2D, Input, AveragePooling2D, Flatten,
                          Dense, Activation, Lambda, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import log_loss, top_k_accuracy_score
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# 텐서보드 사용
from datetime import datetime
root = '/home/ahlee/sichoi/VGG16'
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")

log_dir = f'/home/ahlee/sichoi/VGG16/logs/{timestamp}'
#tensorboard_callback = TensorBoard(log_dir=log_dir,
#                                   histogram_freq=1,
#                                   profile_batch=[500, 520])

logging.set_verbosity(logging.INFO)
AUTOTUNE = tf.data.experimental.AUTOTUNE

# GPU 장치 확인
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))

# 마지막 GPU를 사용하도록 설정
if physical_devices:
    try:
        # 마지막 GPU 선택
        last_gpu = physical_devices[-1]
        tf.config.experimental.set_visible_devices(last_gpu, 'GPU')
        print("Using GPU:", last_gpu)

        # 메모리 사용 동적 조정
        tf.config.experimental.set_memory_growth(last_gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available.")

# 모든 GPU 메모리 사용 허용
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class RandomColorAffine(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomColorAffine, self).__init__(**kwargs)

    def call(self, inputs):
        # RGB -> HSV
        hsv_image = tf.image.rgb_to_hsv(inputs)

        # 랜덤 색상 변화
        hue_delta = tf.random.uniform(shape=[], minval=-0.0005, maxval=0.0005)
        hsv_image = tf.image.adjust_hue(hsv_image, hue_delta)

        # HSV -> RGB
        output = tf.image.hsv_to_rgb(hsv_image)

        return output


    def apply_color_affine(self, inputs, hue_delta, saturation_factor, brightness_delta):
        hsv_image = tf.image.rgb_to_hsv(inputs)
        # hue delta만 변경
        hsv_image = tf.image.adjust_hue(hsv_image, hue_delta)
        hsv_image = tf.image.adjust_saturation(hsv_image, saturation_factor)
        hsv_image = tf.image.adjust_brightness(hsv_image, brightness_delta)
        outputs = tf.image.hsv_to_rgb(hsv_image)
        return outputs


data_preprocessing = tf.keras.Sequential([
    preprocessing.Rescaling(1./255., 1./255.),  # Rescale pixel values
    preprocessing.RandomCrop(224, 224),  # Random crop images to (224, 224)
    preprocessing.RandomFlip(mode='horizontal'),  # Randomly flip images horizontally
    #preprocessing.RandomContrast(factor=0.001),  # Randomly adjust contrast
    #RandomColorAffine(),
])

# 데이터 로드
batch_size = 32

#root2 = '/home/ivpl-d29/dataset/imagenet/'
#root2 = '/home/ivpl-d29/dataset/imagenet_mini/'


from keras.datasets import cifar10
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Split the data into training and validation sets using train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

def preprocess_images(x, y):
    x = tf.image.resize(x, (256, 256))
    return x, y

batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess_images).batch(batch_size).prefetch(AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(preprocess_images).batch(batch_size).prefetch(AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess_images).batch(batch_size).prefetch(AUTOTUNE)


data_preprocessing = tf.keras.Sequential([
    preprocessing.Rescaling(1./255.),  # 픽셀 값을 재조정
    preprocessing.RandomCrop(224, 224),  # 이미지를 (224, 224)로 무작위 자르기
    preprocessing.RandomFlip(mode='horizontal'),  # 이미지를 수평으로 무작위 뒤집기
    #preprocessing.RandomContrast(factor=0.001),  # 대비를 무작위로 조정
    RandomColorAffine(),  # RandomColorAffine 층 추가
])

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (data_preprocessing(x, training=False), y), num_parallel_calls=AUTOTUNE)
def subtract_mean(image):
    image = tf.cast(image, tf.float32)
    mean = tf.reduce_mean(image, axis=[1, 2], keepdims=True)
    centered_image = image - mean
    return centered_image
train_ds = train_ds.map(lambda x, y: (subtract_mean(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (subtract_mean(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (subtract_mean(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)


l2 = tf.keras.regularizers.L2(l2=5e-4)
rd_normal = tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)
bias_0 = tf.keras.initializers.Zeros()

model = Sequential()
model.add(layers.InputLayer(input_shape=(224, 224, 3)))
model.add(Conv2D(64,(3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(MaxPooling2D((2, 2), strides=(2, 2),))
# Block 2
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(MaxPooling2D((2, 2), strides=(2, 2),))
# Block 3
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(MaxPooling2D((2, 2), strides=(2, 2),))
# Block 4)
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(MaxPooling2D((2, 2), strides=(2, 2),))
# Block 5)
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=(1, 1), kernel_initializer='glorot_normal', bias_initializer=bias_0, kernel_regularizer=l2))
model.add(MaxPooling2D((2, 2), strides=(2, 2),))
# FC
model.add(Flatten())
model.add(Dense(4096, activation="relu", kernel_initializer='glorot_normal',))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu", kernel_initializer='glorot_normal',))
model.add(Dropout(0.5))
model.add(Dense(1000, activation="softmax", ))
model.build()
model.summary()

###################################################################
# top-K
tk_callback = tf.metrics.TopKCategoricalAccuracy(k=5)
sparse_tk_callback = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto', min_lr=0.0001, verbose=1)  # patience epoch이내에 줄어들지 않으면 factor만큼 감소
SGD = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, weight_decay=5e-4)
model.compile(optimizer=SGD, loss=losses.categorical_crossentropy, metrics=['accuracy', tk_callback])

# 체크포인트
checkpoint_path = root+"/ckpt/"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)

epochs = 20

'''
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=batch_size,
                    callbacks=[reduce_lr_callback, cp_callback, tensorboard_callback],
                    verbose=1)
score = model.evaluate(test_ds, verbose=1)
'''
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=batch_size,
                    callbacks=[reduce_lr_callback, cp_callback, ],
                    verbose=1)
score = model.evaluate(test_ds, verbose=1)

# 모델 세이브
model.save(os.path.join(checkpoint_path, 'model.h5'))
# 가중치 세이브
model.save_weights(os.path.join(checkpoint_path, 'weights.h5'))

print('Test loss :', score[0])
print('Test accuracy :', score[1])
print(score[0:])
print('Test top-1 error rate: ', 1-score[1])
print('Test top-5 error rate: ', 1-score[2])

# 그래프 그리기 - accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'val'])
plt.title('Accuracy')
# 그래프 그리기 - loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.title('Loss')
plt.show()

# 세션 Clear, 이전 메모리 반환
import keras.backend as K
K.clear_session()