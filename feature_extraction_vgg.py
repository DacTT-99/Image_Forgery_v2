
from keras import applications
import numpy as np


def main():
    model = applications.VGG16(weights='imagenet', include_top=False)

    x_train=np.load('/content/x_trai_data.npy')
    x_test=np.load('/content/x_test_data.npy')

    x_train_bottleneck = model.predict(x_train)
    x_test_bottleneck = model.predict(x_test)

    np.save('x_train_vgg_bottleneck.npy', x_train_bottleneck)
    np.save('x_test_vgg_bottleneck.npy', x_test_bottleneck)

    print(x_train_bottleneck.shape)


if __name__=='__main__':
    main()