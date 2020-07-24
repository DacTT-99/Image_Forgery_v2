import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model,Sequential
from keras.layers import Conv2D,concatenate,Lambda,Input,multiply,add,ZeroPadding2D,Activation,Layer,MaxPool2D,Dropout,BatchNormalization,Flatten,Dense,Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback
from model import *

# create the dir for save model, modifi path to data

init_lr = 0.0001
batch_size = 64
max_epochs = 100
path_resnet = 'tmp/top_model_resnet50'
path_vgg = 'tmp/top_model_vgg16'
period = 5


def lr_decay(epoch):
    return init_lr / (epoch ** 0.5)

def main():

    top_model = create_model_resnet50()

    x_train = np.load('content/x_train_resnet_bottleneck.npy')
    y_train = np.load('content/y_train.npy')

    x_valid = np.load('content/x_test_resnet_bottleneck.npy')
    y_valid = np.load('content/y_test.npy')

    lr_scheduler = LearningRateScheduler(lr_decay)
    ckpt = ModelCheckpoint(filepath=path_resnet + '/top_model_resnet-{epoch:02d}.h5',verbose=1,save_best_only=True,period=period)

    callbacks = [lr_scheduler,ckpt]
    history = top_model.fit(x_train,y_train,batch_size=batch_size,epochs=max_epochs,verbose=1,callbacks=callbacks,validation_data=[x_valid,y_valid],shuffle=True)
