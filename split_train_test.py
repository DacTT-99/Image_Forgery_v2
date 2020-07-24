import cv2
import numpy as np
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
    sample_pristine = []
    for fn in fns:
        img = cv2.imread(fn + '.png')
        for s in sample_random(img,samples_per_img):
            sample_pristine.append(s)

    sample_pristine_np = np.array(sample_pristine)

    print('sample_pristine_np : {}'.format(sample_pristine_np.shape))
    np.save('sample_pristine.npy', sample_pristine_np)

    sample_fake_np = np.load('/content/sample_fake.npy')

    # 20% for valid , 80% for train
    train_fake = sample_fake_np.shape[0] * 4 // 5
    test_fake = sample_fake_np.shape[0] - train_fake
    x_train_fake = sample_fake_np[:train_fake, :, :, :]
    x_test_fake = sample_fake_np[train_fake:, :, :, :]

    # 20% for valid , 80% for train
    train_pristine = sample_pristine_np.shape[0] * 4 // 5
    test_pristine = sample_pristine_np.shape[0] - train_pristine
    x_train_pristine = sample_pristine_np[:train_pristine, :, :, :]
    x_test_pristine = sample_pristine_np[train_pristine:, :, :, :]

    x_train = np.concatenate((x_train_fake, x_train_pristine), axis=0)
    x_test = np.concatenate((x_test_fake, x_test_pristine), axis=0)
    y_train = np.array([0] * train_fake + [1] * train_pristine)
    y_test = np.array([0] * test_fake + [1] * test_pristine)

    np.save('x_train.npy',x_train)
    np.save('x_test.npy',x_test)
    np.save('y_train.npy',y_train)
    np.save('y_test.npy',y_test)

if __name__ == '__main__':
    main()