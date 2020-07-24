import numpy as np

init_lr = 0.00001
batch_size = 64
max_epochs = 100
path_resnet = 'tmp/top_model_resnet50'
path_vgg = 'tmp/top_model_vgg16'
period = 1

def recover_model():

    #rebuild model



    #load weight trained

    #unfreezing

    pass

def lr_decay(epoch):
    return init_lr / (epoch ** 0.5)

def main():
    recover_model()

    x_train = np.load('')
    y_train = np.load('')

    x_valid = np.load('')
    y_valid = np.load('')

    lr_scheduler = LearningRateScheduler(lr_decay)
    ckpt = ModelCheckpoint(filepath=path_resnet + '/top_model_resnet-{epoch:02d}.h5',verbose=1,save_best_only=True,period=period)

    callbacks = [lr_scheduler,ckpt]
    history = top_model.fit(x_train,y_train,batch_size=batch_size,epochs=max_epochs,verbose=1,callbacks=callbacks,validation_data=[x_valid,y_valid],shuffle=True)
