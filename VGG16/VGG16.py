import os

import keras.metrics
import tensorflow as tf
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
root = '/home/ivpl-d29/myProject/Study_Model/VGG16'
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")

log_dir = f'/home/ivpl-d29/myProject/Study_Model/VGG16/logs/{timestamp}'
tensorboard_callback = TensorBoard(log_dir=log_dir,
                                   histogram_freq=1,
                                   profile_batch=[500, 520])

logging.set_verbosity(logging.INFO)
AUTOTUNE = tf.data.experimental.AUTOTUNE

# GPU 장치 확인
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))

# 필요한 경우 GPU 메모리 사용 동적 조정
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 모든 GPU 메모리 사용 허용
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class RandomColorAffine(layers.Layer):
    def __init__(self, **kwargs):
        super(RandomColorAffine, self).__init__(**kwargs)

    def call(self, inputs):
        hsv_image = tf.image.rgb_to_hsv(inputs)
        hue_delta = tf.random.uniform(shape=[], minval=-0.0005, maxval=0.0005)
        hsv_image = tf.image.adjust_hue(hsv_image, hue_delta)
        output = tf.image.hsv_to_rgb(hsv_image)
        return output

    def apply_color_affine(self, inputs, hue_delta, saturation_factor, brightness_delta):
        hsv_image = tf.image.rgb_to_hsv(inputs)
        hsv_image = tf.image.adjust_hue(hsv_image, hue_delta)
        hsv_image = tf.image.adjust_saturation(hsv_image, saturation_factor)
        hsv_image = tf.image.adjust_brightness(hsv_image, brightness_delta)
        outputs = tf.image.hsv_to_rgb(hsv_image)
        return outputs

    # @staticmethod
    # def preprocess_input(x):
    #     # Convert RGB to BGR
    #     x = tf.reverse(x, axis=[-1])
    #     # Zero-center by mean pixel
    #     mean = [103.939, 116.779, 123.68]  # Mean values for BGR channels
    #     x -= mean
    #     return x

def random_resize_train(img):
    scale_factor = tf.random.uniform(shape=[], minval=256, maxval=512, dtype=tf.int32)
    img = tf.image.resize(img, (scale_factor, scale_factor))
    return img

def preprocess_image(image):
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image

# Define data preprocessing pipeline
resize_size = 256
crop_size = 224
mean = [0.485, 0.456, 0.406]
std = tf.constant([0.229, 0.224, 0.225])
variance = tf.square(std)  # std의 각 원소를 제곱하여 분산을 계산

data_preprocessing = tf.keras.Sequential([
    #preprocessing.Resizing(resize_size, resize_size),  # Resize images
    preprocessing.RandomCrop(crop_size, crop_size),  # Center crop to (224, 224)
    preprocessing.Rescaling(1./255),  # Rescale pixel values to [0,1]
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomRotation(factor=0.15),  # 랜덤 회전 (±15도)
    preprocessing.RandomFlip('vertical'),
    preprocessing.Normalization(mean=mean, variance=variance),  # Normalize with mean and std
])

# 데이터 로드
batch_size = 64

root2 = '/home/ivpl-d29/dataset/imagenet/'
#root2 = '/home/ivpl-d29/dataset/imagenet_mini/'
imagenet_train_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(root2+'train'), batch_size=batch_size, shuffle=True, seed=42, image_size=(256, 256),
)
imagenet_val_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(root2+'val'), batch_size=batch_size, shuffle=True, seed=42, image_size=(256, 256),
)
imagenet_test_data = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(root2+'test'), batch_size=batch_size, shuffle=False, seed=42, image_size=(256, 256),
)

def preprocess_dataset(dataset):
    dataset = dataset.map(lambda x, y: (data_preprocessing(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.map(lambda x, y: (preprocess_image(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

train_ds = preprocess_dataset(imagenet_train_data)
val_ds = preprocess_dataset(imagenet_val_data)
test_ds = preprocess_dataset(imagenet_test_data)

print(train_ds)
print(val_ds)
print(test_ds)

# ------------------------------------------------------------------------
rd_normal = tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)
bias_0 = tf.keras.initializers.Zeros()
l2 = tf.keras.regularizers.L2(l2=5e-4)

# --------------------------------------
# 기본 (참고: https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
model = Sequential()
model.add(layers.InputLayer(input_shape=(224, 224, 3)))
model.add(Conv2D(64,  (3, 3),  activation="relu", padding="same", ))
model.add(Conv2D(64,  (3, 3),  activation="relu", padding="same", ))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Block 2
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", ))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Block 3)
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", ))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Block 4))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Block 5))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# FC
model.add(Flatten())
model.add(Dense(4096, activation="relu", ))
model.add(Dense(4096, activation="relu", ))
model.add(Dense(1000, activation="softmax", ))

model.summary()

###################################################################
# top-K
tk_callback = tf.metrics.TopKCategoricalAccuracy(k=5)
sparse_tk_callback = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, mode='auto', min_lr=0.0001, verbose=1)  # patience epoch이내에 줄어들지 않으면 factor만큼 감소
SGD = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
#SGD = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, weight_decay=5e-4,)
model.compile(optimizer=SGD, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy', sparse_tk_callback])

# 체크포인트
save_path = root+"/ckpt/"
checkpoint_path = root+"/ckpt/ckptdir/"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=False, verbose=1)

epochs = 10


history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=batch_size, verbose=1,
                    callbacks=[reduce_lr_callback, cp_callback, tensorboard_callback], )
score = model.evaluate(test_ds, verbose=1)

# 모델 세이브
model.save(os.path.join(save_path, 'model.h5'))
# 가중치 세이브
model.save_weights(os.path.join(save_path, 'weights.h5'))

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
