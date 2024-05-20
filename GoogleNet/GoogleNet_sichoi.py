import os
import tensorflow as tf
import tensorflow.python.keras as keras
from absl import logging
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
import numpy as np
from keras import losses, layers
from keras.models import Sequential, Model
from keras.layers import (Conv2D, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D,
                          MaxPool2D, Input, AveragePooling2D, Flatten,
                          Dense, Activation, Lambda, Dropout, BatchNormalization, concatenate)
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import log_loss, top_k_accuracy_score
tf.executing_eagerly()
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# 텐서보드 사용
from datetime import datetime
root = '/home/ivpl-d29/myProject/'
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")

log_dir = os.path.join(root, f'logs/fit/{timestamp}')
tensorboard_callback = TensorBoard(log_dir=log_dir,
                                   histogram_freq=1,
                                   profile_batch=[500, 505])

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

# 데이터 로드
batch_size = 32

root2 = '/mnt/raid10/ahlee/dataset/imagenet/'
imagenet_train_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(root2+'train'), batch_size=batch_size, image_size=(256,256), shuffle=True)
imagenet_val_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(root2+'val'), batch_size=batch_size, image_size=(256,256), shuffle=True)
imagenet_test_data = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(root2+'test'), batch_size=batch_size, image_size=(256,256), shuffle=False)

data_augmentation = tf.keras.Sequential([
    preprocessing.Resizing(224, 224),
    preprocessing.Rescaling(1./255, 1./255)
])

train_ds = imagenet_train_data.map(lambda x, y: (data_augmentation(x), y))
val_ds = imagenet_val_data.map(lambda x, y: (data_augmentation(x), y))
test_ds = imagenet_test_data.map(lambda x, y: (data_augmentation(x), y))

print(train_ds)
print(val_ds)
print(test_ds)

def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
    # 1st path:
    path1 = Conv2D(filters=f1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # 2nd path
    path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3), padding='same', activation='relu')(path2)

    # 3rd path
    path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5), padding='same', activation='relu')(path3)

    # 4th path
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=f4, kernel_size=(1, 1), padding='same', activation='relu')(path4)

    output_layer = concatenate([path1, path2, path3, path4], axis=-1)

    return output_layer


def GoogLeNet():
    # input layer
    input_layer = Input(shape=(224, 224, 3))

    # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation='relu')(input_layer)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)

    # convolutional layer: filters = 64, strides = 1
    X = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(X)

    # convolutional layer: filters = 192, kernel_size = (3,3)
    X = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(X)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)

    # 1st Inception block
    X = Inception_block(X, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32)

    # 2nd Inception block
    X = Inception_block(X, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)

    # 3rd Inception block
    X = Inception_block(X, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64)

    # 4th Inception block
    X = Inception_block(X, f1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4=64)

    # 5th Inception block
    X = Inception_block(X, f1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4=64)

    # 6th Inception block
    X = Inception_block(X, f1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4=64)

    # 7th Inception block
    X = Inception_block(X, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32,
                        f3_conv5=128, f4=128)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)

    # 8th Inception block
    X = Inception_block(X, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)

    # 9th Inception block
    X = Inception_block(X, f1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48, f3_conv5=128, f4=128)

    # Global Average pooling layer
    X = GlobalAveragePooling2D(name='GAPL')(X)
    # Dropoutlayer
    X = Dropout(0.4)(X)
    # output layer
    X = Dense(10, activation='softmax')(X)

    # model
    model = Model(input_layer, [X], name='GoogLeNet')

    return model
model = GoogLeNet()
model.summary()

tk_callback = tf.metrics.TopKCategoricalAccuracy(k=5)
sparse_tk_callback = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,)
def scheduler(epoch, lr):
    if epoch % 8 == 0 and epoch != 0:
        lr = lr * 0.96  # 4% 감소
    return lr
reduce_lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

model.compile(optimizer=sgd, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy', sparse_tk_callback])

# 체크포인트
checkpoint_path = root+"ckpt/"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

epochs = 50

history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=batch_size,
                    callbacks=[reduce_lr_callback, cp_callback],
                    #callbacks=[reduce_lr_callback, tensorboard_callback, cp_callback],
                    #callbacks=[reduce_lr_callback],
                    verbose=1)
# 모델 세이브
model.save(os.path.join(checkpoint_path, 'model.h5'))
print('모델저장')
# 가중치 세이브
model.save_weights(os.path.join(checkpoint_path, 'weights.h5'))
print('가중치 저장')

score = model.evaluate(test_ds, verbose=1)

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
