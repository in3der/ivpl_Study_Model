import tensorflow as tf
from tensorflow import keras
from keras import losses, layers
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Activation, Lambda, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import log_loss, top_k_accuracy_score
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
import os
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from absl import logging

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# 텐서보드 사용
from datetime import datetime
root = '/home/ahlee/sichoi/AlexNet/'
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")

log_dir = os.path.join(root, f'logs/{timestamp}')
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
    preprocessing.Resizing(256, 256),
    preprocessing.Rescaling(1./255, 1./255)
])

data_augmentation = keras.Sequential(
    [
        preprocessing.Resizing(256,256),
        preprocessing.CenterCrop(227,227),
        preprocessing.RandomFlip(mode='horizontal'),
        preprocessing.Rescaling(1./255, 1./255)
    ]
)

# 데이터 로드
root2 = '/mnt/raid10/ahlee/dataset/imagenet/'
imagenet_train_data = keras.utils.image_dataset_from_directory(os.path.join(root2+'train'))
imagenet_val_data = keras.utils.image_dataset_from_directory(root2+'val')
imagenet_test_data = keras.utils.image_dataset_from_directory(root2+'test')

train_ds = imagenet_train_data.map(lambda x, y: (data_augmentation(x), y))
val_ds = imagenet_val_data.map(lambda x, y: (data_augmentation(x,), y))
test_ds = imagenet_test_data.map(lambda x, y: (data_augmentation(x,), y))

# Display information about the datasets
print(train_ds)
print(val_ds)
print(test_ds)

## 모델 ---------------------------------------
model=Sequential()
model.add(layers.InputLayer(input_shape=(227, 227, 3)))

model.add(Conv2D(96, (11, 11), strides=4, padding='valid', activation='relu'))
model.add(Lambda(tf.nn.local_response_normalization))
model.add(MaxPooling2D(3, strides=2))

model.add(Conv2D(256, (5, 5), strides=1, padding='same', activation='relu'))
model.add(Lambda(tf.nn.local_response_normalization))
model.add(MaxPooling2D(3, strides=2))

model.add(Conv2D(384, (3, 3), strides=1, padding='same', activation='relu'))

model.add(Conv2D(384, (3, 3), strides=1, padding='same', activation='relu'))

model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(3, strides=2))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

model.summary()
###################################################################
# top-K
sparse_tk_callback = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=0.0001)
sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
model.compile(optimizer=sgd, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy', sparse_tk_callback])

# lr감소 추적
def print_lr(epoch, logs):
    current_lr = float(tf.keras.backend.get_value(model.optimizer.lr))
    print(f"Epoch {epoch + 1}/{epochs}, Learning Rate: {current_lr}")
print_lr_callback = tf.keras.callbacks.LambdaCallback(on_epoch_begin=print_lr)

# 체크포인트
checkpoint_path = root+"ckpt/"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

epochs = 90
batch_size = 128
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=batch_size,
                    callbacks=[reduce_lr_callback, tensorboard_callback, cp_callback, print_lr_callback],
                    verbose=1)

# 모델 세이브
model.save(os.path.join(checkpoint_path, 'model.h5'))
# 가중치 세이브
model.save_weights(os.path.join(checkpoint_path, 'weights.h5'))

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