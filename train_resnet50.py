import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model,Sequential
from keras.layers import Conv2D,concatenate,Lambda,Input,multiply,add,ZeroPadding2D,Activation,Layer,MaxPool2D,Dropout,BatchNormalization,Flatten,Dense,Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback
from model import *

init_lr = 0.0001
batch_size = 32
max_epochs = 100
class customModelCheckPoint(Callback):
    def __init__(self, model, path, period, save_weight_only):
        super(customModelCheckPoint,self).__init__()
        self.period = period
        self.model_for_saving = model
        self.save_weight_only = save_weight_only

        self.epchos_since_last_save = 0
    def on_epoch_end(self, epoch, logs=None):
        self.epchos_since_last_save += 1
        if self.epochs_since_last_save  >= self.period:
            self.epochs_since_last_save = 0
            if self.save_weight_only :
                self.model_for_saving.save_weights(self.path.format(epoch=epoch + 1, **logs),overwrite=True)
            else:
                self.model_for_saving.save(self.path.format(epoch=epoch + 1, **logs),overwrite=True)

def lr_decay(epoch):
    return init_lr / (epoch ** 0.5)

def main():

    top_model = create_model_resnet50()

    x_train = np.load('')
    y_train = np.load('')

    x_valid = np.load('')
    y_valid = np.load('')

    lr_scheduler = LearningRateScheduler(lr_decay)

    callbacks = []
    history = top_model.fit(x_train,y_train,batch_size=batch_size,epochs=max_epochs,verbose=1,callbacks=callbacks,validation_data=[x_valid,y_valid],shuffle=True)

    history.history

    # continue wrirting train_resnet50(callbacks), eval.py, and try some other method