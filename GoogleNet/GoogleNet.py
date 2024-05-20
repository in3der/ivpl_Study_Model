# # 세션 Clear, 이전 메모리 반환
# import keras.backend as K
# K.clear_session()

# 필수 라이브러리 및 모듈 임포트
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from keras import losses, layers
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, Input, AveragePooling2D, MaxPooling2D, Flatten, Dense, Activation, Lambda, Dropout, Conv2D, Concatenate, BatchNormalization, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import os
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

# 텐서보드 사용
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=f'/home/ivpl-d29/myProject/logs/fit/{timestamp}', histogram_freq=1)

tf.config.run_functions_eagerly(True)

# GPU 확인
print(device_lib.list_local_devices())
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

batch_size = 64
img_height = 224
img_width = 224
autotune = tf.data.experimental.AUTOTUNE
mean = [0.485, 0.456, 0.406]  # Specify the mean value for normalization
std = [0.229, 0.224, 0.225]   # Specify the standard deviation for normalization

# 참고자료 https://velog.io/@dust_potato/tensorflow2-data-pipeline-multi-threading-using-.map-function
train_path = "/home/ivpl-d29/dataset/imagenet/train"
val_path = "/home/ivpl-d29/dataset/imagenet/val"
test_path = "/home/ivpl-d29/dataset/imagenet/test"
CLASS_NAMES = np.array(sorted(os.listdir(train_path)))

train_list_ds = tf.data.Dataset.list_files(str(train_path + '/*/*'), shuffle=True)
val_list_ds = tf.data.Dataset.list_files(str(val_path + '/*/*'), shuffle=True)
test_list_ds = tf.data.Dataset.list_files(str(test_path + '/*/*'), shuffle=True)

# Define image decoding function
def tf_image_decoding(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    return image, path

# Define preprocessing functions
def tf_preprocess(image, path):
    # Your preprocessing code
    image = tf.cast(image, tf.float32)
    image = ((image / 255.0) - mean) / std
    return image, path

# Define a function to get the label from the path and map it to an integer
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return tf.argmax(parts[-2] == CLASS_NAMES)

# Define a function to create and prepare a dataset
def prepare_dataset(file_paths, batch_size, repeat_count=128, buffer_size=autotune):
    ds = file_paths.interleave(
        lambda file_path: tf.data.Dataset.from_tensor_slices([file_path]),
        cycle_length=1000,
        num_parallel_calls=autotune
    )
    ds = ds.prefetch(buffer_size)
    ds = ds.map(lambda path: tf_image_decoding(path), num_parallel_calls=autotune)
    ds = ds.map(tf_preprocess, num_parallel_calls=autotune)
    ds = ds.map(lambda image, path: (image, tf.cast(get_label(path), tf.int64)), num_parallel_calls=autotune)  # Cast labels to int64
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size)
    return ds

train_ds = prepare_dataset(train_list_ds, batch_size, repeat_count=128, buffer_size=autotune)
val_ds = prepare_dataset(val_list_ds, batch_size, repeat_count=128, buffer_size=autotune)
test_ds = prepare_dataset(test_list_ds, batch_size, repeat_count=128, buffer_size=autotune)


# Display information about the datasets
print(train_ds)
print(val_ds)
print(test_ds)

def visualize_dataset(dataset, num_samples=2, margin=1):
    iterator = iter(dataset)
    for _ in range(num_samples):
        img, label = next(iterator)
        # 전처리 전
        plt.subplot(num_samples, 2, 2 * _ + 1)
        plt.title(f'Label : {CLASS_NAMES[label[0]]}\nOriginal Image\nSize: {img[0].shape[0]} x {img[0].shape[1]}')
        plt.imshow(tf.cast(img[0], tf.uint8))
        plt.axis('off')

        # 전처리 후
        plt.subplot(num_samples, 2, 2 * _ + 2)
        plt.title(f'Label : {CLASS_NAMES[label[1]]}\nPreprocessed Image\nSize: {img[1].shape[0]} x {img[1].shape[1]}')
        plt.imshow(tf.cast(img[1], tf.uint8))
        plt.axis('off')

    plt.subplots_adjust(wspace=margin, hspace=margin)
    plt.show()

# 시각화 출력
# print("Sample from Train set:")
# visualize_dataset(train_ds)
# print("Sample from Validation set:")
# visualize_dataset(val_ds)
# print("Sample from Test set:")
# visualize_dataset(test_ds)

## 모델 ---------------------------------------
# def inception(x, filters):
#     # 1x1
#     path1 = Conv2D(filters=filters[0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
#
#     # 1x1->3x3
#     path2 = Conv2D(filters=filters[1][0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
#     path2 = Conv2D(filters=filters[1][1], kernel_size=(3, 3), strides=1, padding='same', activation='relu')(path2)
#
#     # 1x1->5x5
#     path3 = Conv2D(filters=filters[2][0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
#     path3 = Conv2D(filters=filters[2][1], kernel_size=(5, 5), strides=1, padding='same', activation='relu')(path3)
#
#     # 3x3->1x1
#     path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
#     path4 = Conv2D(filters=filters[3], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(path4)
#
#     return Concatenate(axis=-1)([path1, path2, path3, path4])
#
#
# def auxiliary(x, name=None):
#     layer = AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid')(x)
#     layer = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(layer)
#     layer = Flatten()(layer)
#     layer = Dense(units=256, activation='relu')(layer)
#     layer = Dropout(0.4)(layer)
#     layer = Dense(units=1000, activation='softmax', name=name)(layer)   # units = Class number
#     return layer
#
#
# def googlenet():
#     layer_in = Input(input_shape=(224, 224, 3))
#
#     # stage-1
#     layer = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', activation='relu')(layer_in)
#     layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
#     layer = BatchNormalization()(layer)
#
#     # stage-2
#     layer = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(layer)
#     layer = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(layer)
#     layer = BatchNormalization()(layer)
#     layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
#
#     # stage-3
#     layer = inception(layer, [64, (96, 128), (16, 32), 32])  # 3a
#     layer = inception(layer, [128, (128, 192), (32, 96), 64])  # 3b
#     layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
#
#     # stage-4
#     layer = inception(layer, [192, (96, 208), (16, 48), 64])  # 4a
#     aux1 = auxiliary(layer, name='aux1')
#     layer = inception(layer, [160, (112, 224), (24, 64), 64])  # 4b
#     layer = inception(layer, [128, (128, 256), (24, 64), 64])  # 4c
#     layer = inception(layer, [112, (144, 288), (32, 64), 64])  # 4d
#     aux2 = auxiliary(layer, name='aux2')
#     layer = inception(layer, [256, (160, 320), (32, 128), 128])  # 4e
#     layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
#
#     # stage-5
#     layer = inception(layer, [256, (160, 320), (32, 128), 128])  # 5a
#     layer = inception(layer, [384, (192, 384), (48, 128), 128])  # 5b
#     layer = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(layer)
#
#     # stage-6
#     layer = Flatten()(layer)
#     layer = Dropout(0.4)(layer)
#     layer = Dense(units=256, activation='linear')(layer)
#     main = Dense(units=1000, activation='softmax', name='main')(layer)
#
#     model = Model(inputs=layer_in, outputs=[main, aux1, aux2])
#
#     return model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model

def inception_module(x, filters):
    conv1x1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    conv3x3_reduce = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    conv3x3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(conv3x3_reduce)

    conv5x5_reduce = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    conv5x5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(conv5x5_reduce)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(pool_proj)

    inception = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, pool_proj])
    return inception

def googlenet():
    input_layer = Input(shape=(224, 224, 3))

    conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    avgpool1 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(conv1)

    conv2_reduce = Conv2D(64, (1, 1), padding='same', activation='relu')(avgpool1)
    conv2 = Conv2D(192, (3, 3), padding='same', activation='relu')(conv2_reduce)
    avgpool2 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(conv2)

    inception3a = inception_module(avgpool2, [64, 96, 128, 16, 32, 32])
    inception3b = inception_module(inception3a, [128, 128, 192, 32, 96, 64])
    avgpool3 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(inception3b)

    inception4a = inception_module(avgpool3, [192, 96, 208, 16, 48, 64])
    inception4b = inception_module(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception_module(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception_module(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception_module(inception4d, [256, 160, 320, 32, 128, 128])
    avgpool4 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(inception4e)

    inception5a = inception_module(avgpool4, [256, 160, 320, 32, 128, 128])
    inception5b = inception_module(inception5a, [384, 192, 384, 48, 128, 128])
    avgpool5 = AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(inception5b)
    dropout = Dropout(0.4)(avgpool5)

    flatten = Flatten()(dropout)
    fc1 = Dense(512, activation='relu')(flatten)
    fc2 = Dense(128, activation='relu')(fc1)

    main_output = Dense(6, activation='softmax', name='main_bell')(fc2)
    bell1_output = Dense(6, activation='softmax', name='bell1')(fc2)
    bell2_output = Dense(6, activation='softmax', name='bell2')(fc2)

    model = Model(inputs=input_layer, outputs=[main_output, bell1_output, bell2_output])
    return model


# train model
model = googlenet()
model.summary()

###################################################################

# The learning rate was initialized at 0.01 and reduced three times prior to termination. 3번 나눔이 발생함
# lr 을 1/10씩 나누는데 validation error이 stopped improving할 때 나눔.

sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
def scheduler(epoch, lr):
    if epoch % 8 == 0 and epoch != 0:
        lr = lr * 0.96  # 4% 감소
    return lr
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

model.compile(optimizer=sgd, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
steps_per_epoch=1231167/(batch_size*4)       # train set개수(1231167) / batch size(128) = 9618
validation_steps=50000/(batch_size*4)        # val set 개수(50000) / batch size(128) = 390
epochs = 10

history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback, lr_scheduler],
                   steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, batch_size=batch_size, verbose=1)
# history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[reduce_lr_callback, lr_scheduler],
#                     steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, batch_size=batch_size, verbose=1)

score = model.evaluate(test_ds, verbose=1, steps=steps_per_epoch)
print('Test loss :', score[0])
print('Test accuracy :', score[1])
print(score[0:])

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
