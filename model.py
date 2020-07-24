import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications import VGG16
from keras.models import Model,Sequential
from keras.layers import Conv2D,concatenate,Lambda,Input,multiply,add,ZeroPadding2D,Activation,Layer,MaxPool2D,Dropout,BatchNormalization,Flatten,Dense,Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback

init_lr = 0.0001

def create_model_resnet50():
    resnet = ResNet50(input_shape = [64,64,3],weights='imagenet',include_top=False,pooling=None)
    top_model = Sequential()
    top_model.add(resnet)
    top_model.add(Flatten())
    top_model.add(Dense(64,activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1,activation='sigmoid'))

    for layer in top_model.layers[0].layers[:171]:
		    layer.trainable=False
        
    otp = Adam(init_lr)

    top_model.compile(optimizer=otp,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    return top_model

def create_model_VGG16():
    vgg = VGG16(input_shape=[64,64,3],include_top=False,weights='imagenet',pooling=None,)
    top_model = Sequential()
    top_model.add(vgg)
    top_model.add(Flatten())
    top_model.add(Dense(64,activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1,activation='sigmoid'))

    for layer in top_model.layers[0].layers[:17]:
		   layer.trainable=False

    otp = Adam(init_lr)

    top_model.compile(optimizer=otp,
                      loss='binary_crosentropy',
                      metrics=['accuracy'])
    
    return top_model