import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
fake_path = '/content/dataset-dist/phase-01/training/fake'
pristine_path = '/content/dataset-dist/phase-01/training/pristine'


def get_image(path):
    '''
    return a list contains all images filename in direction
    '''
    file = os.listdir(path)
    fn = set()
    for i in file:
        if i.split('.')[0] == '':
            continue
        fn.add(path + '/' + i.split('.')[0])
    return list(fn)


def count_fake_point(mask):
    '''
    return number of point with values equal to 255 (white point) in mask
    '''
    return mask[mask == 255].shape[0]


def sample_fake(img, mask):
    kernel_size = 64
    stride = 8
    threshold = 1600
    samples = []
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    for y_start in range(0, img.shape[0] - kernel_size + 1, stride):
        for x_start in range(0, img.shape[1] - kernel_size + 1, stride):

            fake_point = count_fake_point(
                mask[y_start:y_start + kernel_size, x_start:x_start + kernel_size])

            if (fake_point > threshold) and (kernel_size * kernel_size - fake_point > threshold):
                samples.append(
                    img[y_start:y_start + kernel_size, x_start:x_start + kernel_size, :3])

    return samples


def main():
    fns = get_image(fake_path)
    x_train_mask = []
    x_train_fake_images = []
    x_fake_train = []
    x_fake_test = []

    y = np.array([0]*len(fns))
    fns_train ,fns_test,_,_=train_test_split(fns,y,test_size=0.2,stratify=y)

    for fn in fns_train:
        img = cv2.imread(fn + '.png')
        mask = cv2.imread(fn + '.mask.png',0)
        blur = cv2.GaussianBlur(mask,(5,5),0)
        ret3,mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        for s in sample_fake(img, mask):
            x_fake_train.append(s)

    for fn in fns_test:
        img = cv2.imread(fn + '.png')
        mask = cv2.imread(fn + '.mask.png')
        for s in sample_fake(img, mask):
            x_fake_test.append(s)

    x_fake_train = np.array(x_fake_train)
    x_fake_test = np.array(x_fake_test)

    y_fake_train = np.array([0]*x_fake_train.shape[0])
    y_fake_test = np.array([0]*x_fake_test.shape[0])

    # print(samples_fake_np.shape)
    print('done')

    np.save('x_fake_train.npy', x_fake_train)
    np.save('x_fake_test.npy', x_fake_test)
    np.save('y_fake_train.npy', y_fake_train)
    np.save('y_fake_test.npy', y_fake_test)


if __name__ == '__main__':
    main()
