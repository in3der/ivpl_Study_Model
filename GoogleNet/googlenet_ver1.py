#googlenet_ver1

import os
import datetime
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,Concatenate, Dense, Flatten,BatchNormalization, Dropout
from keras.activations import softmax,relu
from keras.losses import MSE, categorical_crossentropy
from keras.optimizers import SGD
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras.callbacks import Callback

#데이터셋 설정
train_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.2,height_shift_range =0.2,
    zoom_range=0.2, horizontal_flip =True, vertical_flip = True)

train_generator = train_datagen.flow_from_directory(
    '/home/ivpl-d29/dataset/imagenet/train/',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.2,height_shift_range =0.2,
zoom_range=0.2, horizontal_flip =True, vertical_flip = True)

test_generator = test_datagen.flow_from_directory(
    '/home/ivpl-d29/dataset/imagenet/test/',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical')

class googlenet(tf.keras.Model):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True):
        super(googlenet, self).__init__()
        self.first=Sequential([
            Conv2D(filters=64,kernel_size=7,strides=(2,2),activation='relu', input_shape=(224,224,3)),
            MaxPooling2D(pool_size=(3,3),strides=(2,2)),
            BatchNormalization(),
            Conv2D(filters=64,kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=192,kernel_size=3,strides=(1,1),activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(3,3),strides=(2,2))
        ])

        self.inception_3a_1=Sequential([
            Conv2D(filters=64, kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_3a_2=Sequential([
            Conv2D(filters=96, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=128,kernel_size=3,strides=(1,1),activation='relu')
        ])

        self.inception_3a_3=Sequential([
            Conv2D(filters=16, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=32,kernel_size=5,strides=(1,1),activation='relu')
        ])

        self.inception_3a_4=Sequential([
            MaxPooling2D(pool_size=(3,3),strides=(2,2)),
            Conv2D(filters=32,kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.MaxP3=Sequential([
            MaxPooling2D(pool_size=(3,3),strides=(2,2))
        ])

        self.inception_3b_1=Sequential([
            Conv2D(filters=128, kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_3b_2=Sequential([
            Conv2D(filters=128, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=192,kernel_size=3,strides=(1,1),activation='relu')
        ])
        self.inception_3b_3=Sequential([
            Conv2D(filters=32, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=96,kernel_size=5,strides=(1,1),activation='relu')
        ])
        self.inception_3b_4=Sequential([
            MaxPooling2D(pool_size=(3,3),strides=(2,2)),
            Conv2D(filters=64,kernel_size=1,strides=(1,1),activation='relu')
        ])
        self.inception_3b_5=Sequential([
            AveragePooling2D(pool_size=(5,5),strides=(2,2)),
            Conv2D(filters=128,kernel_size=1,strides=(1,1),activation='relu'),
            Dense(1024,activation='relu'),
            Dropout(0.7),
            Dense(1000,activation='softmax')
        ])

        self.inception_4a_1=Sequential([
            Conv2D(filters=192, kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_4a_2=Sequential([
            Conv2D(filters=96, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=208, kernel_size=3,strides=(1,1),activation='relu')
        ])

        self.inception_4a_3=Sequential([
            Conv2D(filters=16, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=48,kernel_size=5,strides=(1,1),activation='relu')
        ])
        self.inception_4a_4=Sequential([
            MaxPooling2D(pool_size=(3,3),strides=(2,2)),
            Conv2D(filters=64,kernel_size=1,strides=(1,1),activation='relu')
        ])
        

        self.inception_4b_1=Sequential([
            Conv2D(filters=160, kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_4b_2=Sequential([
            Conv2D(filters=112, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=224, kernel_size=3,strides=(1,1),activation='relu')
        ])

        self.inception_4b_3=Sequential([
            Conv2D(filters=24, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=64,kernel_size=5,strides=(1,1),activation='relu')
        ])
        self.inception_4b_4=Sequential([
            MaxPooling2D(pool_size=(3,3),strides=(2,2)),
            Conv2D(filters=64,kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_4c_1=Sequential([
            Conv2D(filters=128, kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_4c_2=Sequential([
            Conv2D(filters=128, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=256, kernel_size=3,strides=(1,1),activation='relu')
        ])

        self.inception_4c_3=Sequential([
            Conv2D(filters=24, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=64,kernel_size=5,strides=(1,1),activation='relu')
        ])
        self.inception_4c_4=Sequential([
            MaxPooling2D(pool_size=(3,3),strides=(2,2)),
            Conv2D(filters=64,kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_4d_1=Sequential([
            Conv2D(filters=112, kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_4d_2=Sequential([
            Conv2D(filters=144, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=288, kernel_size=3,strides=(1,1),activation='relu')
        ])

        self.inception_4d_3=Sequential([
            Conv2D(filters=32, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=64,kernel_size=5,strides=(1,1),activation='relu')
        ])
        self.inception_4d_4=Sequential([
            MaxPooling2D(pool_size=(3,3),strides=(2,2)),
            Conv2D(filters=64,kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_4e_1=Sequential([
            Conv2D(filters=256, kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_4e_2=Sequential([
            Conv2D(filters=160, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=320, kernel_size=3,strides=(1,1),activation='relu')
        ])

        self.inception_4e_3=Sequential([
            Conv2D(filters=32, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=128,kernel_size=5,strides=(1,1),activation='relu')
        ])
        self.inception_4e_4=Sequential([
            MaxPooling2D(pool_size=(3,3),strides=(2,2)),
            Conv2D(filters=128,kernel_size=1,strides=(1,1),activation='relu')
        ])
        self.inception_4e_5=Sequential([
            AveragePooling2D(pool_size=(5,5),strides=(2,2)),
            Conv2D(filters=128,kernel_size=1,strides=(1,1),activation='relu'),
            Dense(1024,activation='relu'),
            Dropout(0.7),
            Dense(1000,activation='softmax')
        ])
        self.inception_5a_1=Sequential([
            Conv2D(filters=256, kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_5a_2=Sequential([
            Conv2D(filters=160, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=320, kernel_size=3,strides=(1,1),activation='relu')
        ])

        self.inception_5a_3=Sequential([
            Conv2D(filters=32, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=128,kernel_size=5,strides=(1,1),activation='relu')
        ])
        self.inception_5a_4=Sequential([
            MaxPooling2D(pool_size=(3,3),strides=(2,2)),
            Conv2D(filters=128,kernel_size=1,strides=(1,1),activation='relu')
        ])
        

        self.inception_5b_1=Sequential([
            Conv2D(filters=384, kernel_size=1,strides=(1,1),activation='relu')
        ])

        self.inception_5b_2=Sequential([
            Conv2D(filters=192, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=384, kernel_size=3,strides=(1,1),activation='relu')
        ])

        self.inception_5b_3=Sequential([
            Conv2D(filters=48, kernel_size=1,strides=(1,1),activation='relu'),
            Conv2D(filters=128,kernel_size=5,strides=(1,1),activation='relu')
        ])
        self.inception_5b_4=Sequential([
            MaxPooling2D(pool_size=(3,3),strides=(2,2)),
            Conv2D(filters=128,kernel_size=1,strides=(1,1),activation='relu')
        ])
        
        self.AvgP=Sequential([
            AveragePooling2D(pool_size=(7,7),strides=(1,1))
        ])

        self.Dropout=Sequential([
            Dropout(0.6)
        ])

        self.FC=Sequential([
            Dense(1000,activation='softmax')
        ])

    def call(self, x):
        x=self.first(x)
        x1=self.inception_3a_1(x)
        x2=self.inception_3a_2(x)
        x3=self.inception_3a_3(x)
        x4=self.inception_3a_4(x)
        #depthconcat
        x = Concatenate(axis=-1)([x1, x2, x3, x4])
        #maxpooling
        x=self.MaxP3(x)
        x1=self.inception_3b_1(x)
        x2=self.inception_3b_2(x)
        x3=self.inception_3b_3(x)
        x4=self.inception_3b_4(x)
        soft0=self.inception_3b_5(x)
        #depthconcat
        x = Concatenate(axis=-1)([x1, x2, x3, x4])

        x1=self.inception_4a_1(x)
        x2=self.inception_4a_2(x)
        x3=self.inception_4a_3(x)
        x4=self.inception_4a_4(x)
        
        #depthconcat
        x = Concatenate(axis=-1)([x1, x2, x3, x4])

        x1=self.inception_4b_1(x)
        x2=self.inception_4b_2(x)
        x3=self.inception_4b_3(x)
        x4=self.inception_4b_4(x)
        #depthconcat
        x = Concatenate(axis=-1)([x1, x2, x3, x4])

        x1=self.inception_4c_1(x)
        x2=self.inception_4c_2(x)
        x3=self.inception_4c_3(x)
        x4=self.inception_4c_4(x)

        #depthconcat
        x = Concatenate(axis=-1)([x1, x2, x3, x4])

        x1=self.inception_4d_1(x)
        x2=self.inception_4d_2(x)
        x3=self.inception_4d_3(x)
        x4=self.inception_4d_4(x)

        x = Concatenate(axis=-1)([x1, x2, x3, x4])

        x1=self.inception_4e_1(x)
        x2=self.inception_4e_2(x)
        x3=self.inception_4e_3(x)
        x4=self.inception_4e_4(x)
        soft1=self.inception_4e_5(x)

        #depthconcat
        x = Concatenate(axis=-1)([x1, x2, x3, x4])
        #maxpooling
        x=self.MaxP3(x)

        x1=self.inception_5a_1(x)
        x2=self.inception_5a_2(x)
        x3=self.inception_5a_3(x)
        x4=self.inception_5a_4(x)

        #depthconcat
        x = Concatenate(axis=-1)([x1, x2, x3, x4])

        x1=self.inception_5b_1(x)
        x2=self.inception_5b_2(x)
        x3=self.inception_5b_3(x)
        x4=self.inception_5b_4(x)

        #depthconcat
        x = Concatenate(axis=-1)([x1, x2, x3, x4])

        x=self.AvgP(x)
        x=self.Dropout(x)
        soft2=self.FC(x)

        return x


#main
# class LearningRateSchedule(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         if (epoch+1)%8 == 0:
#             lr = K.get_value(self.model.optimizer.lr)
#             K.set_value(self.model.optimizer.lr, lr*0.96)
model = googlenet()
model.build(input_shape=(None, 224,224,3))



sgd = SGD(learning_rate=0.01,momentum=0.9) #asynchronous SGD
# googlenet.compile(loss={'soft2': 'categorical_crossentropy',
#                          'soft1': 'categorical_crossentropy',
#                          'soft0': 'categorical_crossentropy'},
#                   loss_weights={
#                       'soft2': 1.0,
#                       'soft1': 0.3,
#                       'soft0': 0.3},
#                   optimizer=sgd, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
callback_list = [tensorboard_callback]

model.summary()

model.fit_generator(train_generator, epochs=10,verbose=1,validation_data=test_generator)



#모델 저장하기
model.save('googlenet.h5')

#모델 평가하기
print("-------------Evaluate-----------------")
scores = model.evaluate_generator(test_generator,steps=1)
print("%s : %.2f%%" %(model.metrics_names[1],scores[1]*100))



