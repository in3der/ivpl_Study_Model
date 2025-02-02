import os
import tensorflow as tf
from absl import logging
from tensorflow import keras
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

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# 텐서보드 사용
from datetime import datetime
root = '/home/ivpl-d29/myProject/Study_Model/Inception-v4'
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")

log_dir = os.path.join(root, f'logs/{timestamp}')
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


data_preprocessing = keras.Sequential([
    preprocessing.Resizing(299, 299),
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


def InceptionV4():
    input_layer = Input(shape=(299, 299, 3))

    x = stemBlock(prev_layer=input_layer)

    x = InceptionBlock_A(prev_layer=x)
    x = InceptionBlock_A(prev_layer=x)
    x = InceptionBlock_A(prev_layer=x)
    x = InceptionBlock_A(prev_layer=x)

    x = ReductionBlock_A(prev_layer=x)

    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)

    x = ReductionBlock_B(prev_layer=x)

    x = InceptionBlock_C(prev_layer=x)
    x = InceptionBlock_C(prev_layer=x)
    x = InceptionBlock_C(prev_layer=x)

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=x, name='Inception-v4')
    return model

# conv layer + BN + activation 결합한 작은 블록
def conv2d_with_Batch(prev_layer, nbr_kernels, filter_size=(3,3), strides=(1,1), padding='same'):
    x = Conv2D(filters=nbr_kernels, kernel_size=filter_size, strides=strides, padding=padding) (prev_layer)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def stemBlock(prev_layer):
    x = conv2d_with_Batch(prev_layer, nbr_kernels=32, filter_size=(3,3), strides=(2,2), padding='valid')     # 3X3 Conv (32 stride 2 V) -> 149*149*32
    x = conv2d_with_Batch(x, nbr_kernels=32, filter_size=(3,3), padding='valid')     # 3X3 Conv (32 V) -> 147*147*32
    x = conv2d_with_Batch(x, nbr_kernels=64, filter_size=(3,3))     # 3X3 Conv (64) -> 147*147*64

    x_1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)  # 3X3 MaxPool (stride 2 V)
    x_2 = conv2d_with_Batch(x, nbr_kernels=96, filter_size=(3,3), strides=(2,2), padding='valid')        # 3X3 Conv (96 stride 2 V)
    x = concatenate([x_1, x_2], axis=-1)     # Filter concat -> 73*73*160

    x_1 = conv2d_with_Batch(x, nbr_kernels=64, filter_size=(1,1))       # 1X1 Conv (64)
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=64, filter_size=(7,1))     # 7X1 Conv (64)
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=64, filter_size=(1,7))     # 1X7 Conv (64)

    x_1 = conv2d_with_Batch(x_1, nbr_kernels=96, filter_size=(3,3), padding='valid')     # 3X3 Conv (96 V)
    x_2 = conv2d_with_Batch(x, nbr_kernels=96, filter_size=(1,1))       # 1X1 Conv (64)
    x_2 = conv2d_with_Batch(x_2, nbr_kernels=96, filter_size=(3,3), padding='valid')     # 3X3 Conv (96 V)
    x = concatenate([x_1, x_2], axis=-1)     # Filter concat -> 71*71*192

    x_1 = conv2d_with_Batch(x, nbr_kernels=192, filter_size=(3,3), strides=2, padding='valid')       # 3X3 Conv (192 V)
    x_2 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid') (x)     # MaxPool (stride=2 V)
    x = concatenate([x_1, x_2], axis=-1)     # Filter concat -> 35*35*384

    return x

def InceptionBlock_A(prev_layer):
    x_1 = conv2d_with_Batch(prev_layer, nbr_kernels=64, filter_size=(1,1))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=96, filter_size=(3,3))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=96, filter_size=(3,3))

    x_2 = conv2d_with_Batch(prev_layer, nbr_kernels=64, filter_size=(1,1))
    x_2 = conv2d_with_Batch(x_2, nbr_kernels=96, filter_size=(3,3))

    x_3 = conv2d_with_Batch(prev_layer, nbr_kernels=96, filter_size=(1,1))

    x_4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(prev_layer)
    x_4 = conv2d_with_Batch(x_4, nbr_kernels=96, filter_size=(1,1))

    x = concatenate([x_1, x_2, x_3, x_4], axis=-1)

    return x

def ReductionBlock_A(prev_layer):
    x_1 = conv2d_with_Batch(prev_layer, nbr_kernels=192, filter_size=(1,1))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=224, filter_size=(3,3))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=256, filter_size=(3,3), strides=(2,2), padding='valid')

    x_2 = conv2d_with_Batch(prev_layer, nbr_kernels=384, filter_size=(3,3), strides=(2,2), padding='valid')

    x_3 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')(prev_layer)

    x = concatenate([x_1, x_2, x_3], axis=-1)

    return x

def InceptionBlock_B(prev_layer):
    x_1 = conv2d_with_Batch(prev_layer, nbr_kernels=192, filter_size=(1,1))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=192, filter_size=(1,7))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=224, filter_size=(7,1))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=224, filter_size=(1,7))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=256, filter_size=(7,1))

    x_2 = conv2d_with_Batch(prev_layer, nbr_kernels=192, filter_size=(1,1))
    x_2 = conv2d_with_Batch(x_2, nbr_kernels=224, filter_size=(7,1))
    x_2 = conv2d_with_Batch(x_2, nbr_kernels=256, filter_size=(1,7))

    x_3 = conv2d_with_Batch(prev_layer, nbr_kernels=384, filter_size=(1,1))

    x_4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(prev_layer)
    x_4 = conv2d_with_Batch(x_4, nbr_kernels=128, filter_size=(1,1))

    x = concatenate([x_1, x_2, x_3, x_4], axis=-1)

    return x


def ReductionBlock_B(prev_layer):
    x_1 = conv2d_with_Batch(prev_layer, nbr_kernels=256, filter_size=(1,1))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=256, filter_size=(1,7))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=320, filter_size=(7,1))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=320, filter_size=(3,3), strides=(2,2), padding='valid')

    x_2 = conv2d_with_Batch(prev_layer, nbr_kernels=192, filter_size=(1,1))
    x_2 = conv2d_with_Batch(x_2, nbr_kernels=192, filter_size=(3,3), strides=(2,2), padding='valid')

    x_3 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')(prev_layer)

    x = concatenate([x_1, x_2, x_3], axis=-1)

    return x

def InceptionBlock_C(prev_layer):
    x_1 = conv2d_with_Batch(prev_layer, nbr_kernels=384, filter_size=(1,1))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=448, filter_size=(1,1))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=512, filter_size=(3,1))
    x_1_1 = conv2d_with_Batch(x_1, nbr_kernels=256, filter_size=(1,3))
    x_1_2 = conv2d_with_Batch(x_1, nbr_kernels=256, filter_size=(3,1))

    x_2 = conv2d_with_Batch(prev_layer, nbr_kernels=384, filter_size=(1,1))
    x_2_1 = conv2d_with_Batch(x_2, nbr_kernels=256, filter_size=(3,1))
    x_2_2 = conv2d_with_Batch(x_2, nbr_kernels=256, filter_size=(1,3))

    x_3 = conv2d_with_Batch(prev_layer, nbr_kernels=256, filter_size=(1,1))

    x_4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(prev_layer)
    x_4 = conv2d_with_Batch(x_4, nbr_kernels=256, filter_size=(1,1))

    x = concatenate([x_1_1, x_1_2, x_2_1, x_2_2, x_3, x_4], axis=-1)

    return x

model = InceptionV4()
model.summary()

###################################################################
# top-K
tk_callback = tf.metrics.TopKCategoricalAccuracy(k=5)
initial_learning_rate = 0.045
def lr_schedule(epoch, lr):
    if epoch % 2 == 0:
        return lr * 0.94
    return lr
#reduce_lr_callback = ExponentialDecay(initial_learning_rate=initial_learning_rate, decay_steps=2, decay_rate=0.94)
reduce_lr_callback = LearningRateScheduler(lr_schedule)
# gradient clipping value - 보통 - -1~+1 사이
RMSprop = tf.keras.optimizers.legacy.RMSprop(learning_rate=initial_learning_rate, decay=0.9, epsilon=1.0, clipvalue=1)
model.compile(optimizer=RMSprop, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy', tk_callback])
model.compile(optimizer=RMSprop, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy', tk_callback, 'sparse_top_k_categorical_accuracy'])

# 체크포인트
checkpoint_path = root+"ckpt/"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
epochs = 33
batch_size = 128
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=batch_size,
                    callbacks=[reduce_lr_callback, cp_callback, tensorboard_callback],
                    verbose=1)
score = model.evaluate(test_ds, verbose=1)

# 이동 평균을 위한 객체 생성
ema = tf.train.ExponentialMovingAverage(decay=0.99)
# 훈련 스텝 정의
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.keras.losses.sparse_categorical_crossentropy(targets, predictions, from_logits=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    # 모델 파라미터 업데이트 전에 이동 평균 업데이트
    ema.apply(gradients, model.trainable_variables)
    tf.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
# 예측을 위한 함수 정의
def predict(inputs):
    # 이동 평균 적용된 파라미터로 예측
    ema.assign(model.variables)
    return model(inputs)


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





