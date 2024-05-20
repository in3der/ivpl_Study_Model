# # 세션 Clear, 이전 메모리 반환
# import keras.backend as K
# K.clear_session()
# exit()

import os
import tensorflow as tf
tf.executing_eagerly()
import keras
from absl import logging
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
import numpy as np
from keras import losses, layers
from keras.models import Sequential, Model
from keras.layers import (Conv2D, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D,
                          MaxPool2D, Input, AveragePooling2D, Flatten,
                          Dense, Activation, Lambda, Dropout, BatchNormalization)
from keras.layers import concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import log_loss, top_k_accuracy_score
tf.executing_eagerly()
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

# 텐서보드 사용
from datetime import datetime
root = '/home/ivpl-d29/myProject/'
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")

log_dir = os.path.join(root, f'logs/fit/{timestamp}')
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


data_preprocessing = keras.Sequential([
    preprocessing.Resizing(224, 224),
    preprocessing.Rescaling(1./255, 1./255)
])

# 데이터 로드
root2 = '/home/ivpl-d29/dataset/imagenet/'
imagenet_train_data = keras.preprocessing.image_dataset_from_directory(os.path.join(root2+'train'))
imagenet_val_data = keras.preprocessing.image_dataset_from_directory(root2+'val')
imagenet_test_data = keras.preprocessing.image_dataset_from_directory(root2+'test')

train_ds = imagenet_train_data.map(lambda x, y: (data_preprocessing(x, training=True), y))
val_ds = imagenet_val_data.map(lambda x, y: (data_preprocessing(x, training=False), y))
test_ds = imagenet_test_data.map(lambda x, y: (data_preprocessing(x, training=False), y))

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Display information about the datasets
print(train_ds)
print(val_ds)
print(test_ds)

## 모델 ---------------------------------------
# GPU 1개 사용할 경우
def GoogLeNet():
    input_layer = Input(shape=(224, 224, 3))



    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=x, name='GoogLeNet')
    return model

def inception_Block(prev_layer):
    x = Conv2D(prev_layer, )


    return x

model = GoogLeNet()
model.summary()

###################################################################
# top-K
tk_callback = tf.metrics.TopKCategoricalAccuracy(k=5)

reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=0.0001)
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
model.compile(optimizer=sgd, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy', tk_callback])

# 체크포인트
checkpoint_path = root+"ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

epochs = 90
batch_size = 128
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=batch_size,
                    # callbacks=[reduce_lr_callback, tk_callback, tk_callback, tensorboard_callback, cp_callback],
                    callbacks=[reduce_lr_callback, tk_callback, tk_callback],
                    verbose=1)

score = model.evaluate(test_ds, verbose=1)
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