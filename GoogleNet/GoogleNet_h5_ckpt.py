# 0. 사용할 패키지 불러오기
import tensorflow as tf
import keras
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Model
from keras import losses, layers
from keras.models import Sequential, Model
from keras.layers import (Conv2D, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D,
                          MaxPool2D, Input, AveragePooling2D, Flatten,
                          Dense, Activation, Lambda, Dropout, BatchNormalization, concatenate)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
AUTOTUNE = tf.data.experimental.AUTOTUNE

# 1. 데이터 준비하기
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Split the data into training and validation sets using train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)

def preprocess_images(x, y):
    x = tf.image.resize(x, (224,224))
    x = x / 255.0
    return x, y

batch_size = 16

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess_images).shuffle(buffer_size=10000).batch(batch_size).prefetch(AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(preprocess_images).batch(batch_size).prefetch(AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess_images).batch(batch_size).prefetch(AUTOTUNE)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print(train_ds)
print(val_ds)
print(test_ds)

# --------------------------------------------------------

# model.h5 로드
from tensorflow.keras.models import load_model
model = load_model('/home/ivpl-d29/myProject/ckpt/GoogleNet_cifar10/model.h5')
model.summary()

score = model.evaluate(test_ds, verbose=1)
print('h5 Test loss :', score[0])
print('h5 Test accuracy :', score[1])
print(score[0:])

# 새로운 데이터 불러와 예측
new_image_path = '/home/ivpl-d29/dataset/cifar_test/dog.jpg'  # 새로운 이미지의 경로
img = image.load_img(new_image_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # 이미지를 모델에 입력하기 전에 정규화

# 예측
predictions = model.predict(img_array)

# 예측 percentage 출력 함수 생성
def predict_class_percentage(predictions):
    def convert_to_class(prediction):
        return np.argmax(prediction)

    predicted_class = convert_to_class(predictions[0])
    print(f"Predicted Class: {predicted_class}")

    # 각 클래스에 대한 예측 확률 출력
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    for i in range(len(class_names)):
        print(f"{class_names[i]}: {predictions[0][i] * 100:.2f}%")
    # 이미지 시각화
    plt.imshow(img)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.show()

predict_class_percentage(predictions)
# --------------------------------------------------------

# weight.h5 로드
def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
    path1 = Conv2D(filters=f1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3), padding='same', activation='relu')(path2)
    path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5), padding='same', activation='relu')(path3)
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=f4, kernel_size=(1, 1), padding='same', activation='relu')(path4)
    output_layer = concatenate([path1, path2, path3, path4], axis=-1)
    return output_layer
def GoogLeNet():
    input_layer = Input(shape=(224, 224, 3))
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation='relu')(input_layer)
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
    X = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(X)
    X = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
    X = Inception_block(X, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32)
    X = Inception_block(X, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64)
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
    X = Inception_block(X, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64)
    X = Inception_block(X, f1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4=64)
    X = Inception_block(X, f1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4=64)
    X = Inception_block(X, f1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4=64)
    X = Inception_block(X, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32,
                        f3_conv5=128, f4=128)
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
    X = Inception_block(X, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)
    X = Inception_block(X, f1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48, f3_conv5=128, f4=128)
    X = GlobalAveragePooling2D(name='GAPL')(X)
    X = Dropout(0.4)(X)
    X = Dense(10, activation='softmax')(X)
    model = Model(input_layer, [X], name='GoogLeNet')
    return model

# 모델 지정
new_model = GoogLeNet()
# 가중치 불러오기
new_model.load_weights('/home/ivpl-d29/myProject/ckpt/GoogleNet_cifar10/weights.h5')
# 컴파일 필요
sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
new_model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['accuracy'])

score = new_model.evaluate(test_ds, verbose=1)

print('weight Test loss :', score[0])
print('weight Test accuracy :', score[1])
print(score[0:])

# 새로운 데이터 불러와 예측
new_image_path = '/home/ivpl-d29/dataset/cifar_test/horse.jpg'
img = image.load_img(new_image_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # 이미지를 모델에 입력하기 전에 정규화

# 예측
predictions = new_model.predict(img_array)

# 예측 결과 출력
predict_class_percentage(predictions)

# --------------------------------------------------------

# 체크포인트 로드
# weight.h5에서 불러온 모델 그대로 사용필요
new_model.load_weights('/home/ivpl-d29/myProject/ckpt/GoogleNet_cifar10/')
sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
new_model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['accuracy'])

score = new_model.evaluate(test_ds, verbose=1)

print('ckpt Test loss :', score[0])
print('ckpt Test accuracy :', score[1])
print(score[0:])

# 새로운 데이터 불러와 예측
new_image_path = '/home/ivpl-d29/dataset/cifar_test/ship.jpg'  # 새로운 이미지의 경로
img = image.load_img(new_image_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # 이미지를 모델에 입력하기 전에 정규화

# 예측
predictions = new_model.predict(img_array)

# 예측 결과 출력
predict_class_percentage(predictions)
