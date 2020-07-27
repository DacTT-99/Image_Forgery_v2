import cv2
import numpy as np
from  sklearn.model_selection import train_test_split
from sample_fake_image import *


fake_path = '/content/dataset-dist/phase-01/training/fake'
pristine_path = '/content/dataset-dist/phase-01/training/pristine'

num_fake_samples = 99754        # threshold = 1600 stride = 8
num_pristine_image = 1050

samples_per_img = num_fake_samples // num_pristine_image

def sample_random(img, num_samples):
    '''
    randomly sample the given image n times

    Parameters
    -----------
    img : array
        image need to sample

    num_samples : number of time to sample
    '''
    samples = []
    rol, cow, chanel = img.shape
    if chanel > 3 :
        img = img[:,:,:3]
    for i in range(num_samples):
        x = np.random.randint(0, rol-64)
        y = np.random.randint(0, cow-64)
        samples.append(img[x:x+64, y:y+64, :])

    return samples


def main():
    fns = get_image(pristine_path)
    y = np.array([0]*len(fns))
    fns_train,fns_test,_,_=train_test_split(fns,y,test_size=0.2,stratify=fns)

    x_pristine_train = []
    x_pristine_test = []

    for fn in fns_train:
        img = cv2.imread(fn + '.png')
        for s in sample_random(img,samples_per_img):
            x_pristine_train.append(s)

    for fn in fns_test:
        img = cv2.imread(fn + '.png')
        for s in sample_random(img,samples_per_img):
            x_pristine_test.append(s)

    x_pristine_train = np.array(x_pristine_train)
    x_pristine_test = np.array(x_pristine_test)
    y_pristine_train = np.array([1]*x_pristine_train.shape[0])
    y_pristine_test = np.array([1]*x_pristine_test.shape[0])

    np.save('x_pristine_train.npy', x_pristine_train)
    np.save('x_pristine_test.npy', x_pristine_test)
    np.save('y_pristine_train.npy', y_pristine_train)
    np.save('y_pristine_test.npy', y_pristine_test)

    x_fake_train = np.load('/content/x_fake_train.npy')
    x_fake_test = np.load('/content/x_fake_test.npy')
    y_fake_train = np.load('/content/y_fake_train.npy')
    y_fake_test = np.load('/content/y_fake_test.npy')

    print('Train data : {} pristine samples + {} fake samples'.format(y_pristine_train.shape[0],y_fake_train.shape[0]))
    print('Validation data : {} pristine samples + {} fake samples'.format(y_pristine_test.shape[0],y_fake_test.shape[0]))

    x_train_data = np.concatenate((x_pristine_train,x_fake_train),axis=0)
    x_test_data = np.concatenate((x_pristine_test,x_fake_test),axis=0)
    y_train_data = np.concatenate((y_pristine_train,y_fake_train),axis=0)
    y_test_data = np.concatenate((y_pristine_test,y_fake_test),axis=0)

    np.save('x_train_data.npy',x_train_data)
    np.save('x_test_data.npy',x_test_data)
    np.save('y_train_data.npy',y_train_data)
    np.save('y_test_data.npy',y_test_data)

if __name__ == '__main__':
    main()