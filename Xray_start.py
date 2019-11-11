from os import environ, chdir
import scipy.ndimage as sp
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers, callbacks
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
chdir(r'C:\skr\xray')
train_images_normal = os.listdir(r'data1\train\NORMAL')
train_images_xray =os.listdir(r'data1\train\PNEUMONIA')
test_images_normal = os.listdir(r'data1\test\NORMAL')
test_images_xray =os.listdir(r'data1\test\PNEUMONIA')
val_images_normal = os.listdir(r'data1\val\NORMAL')
val_images_xray =os.listdir(r'data1\val\PNEUMONIA')
print('X-RAY data-set size')
print('No. of images in Train data-set=',
len(train_images_xray)+len(train_images_normal))
print('No. of images in Test data-set=',
len(test_images_xray)+len(test_images_normal))
print('No. of images in Valid data-set=',
len(val_images_xray)+len(val_images_normal))
print('Normal Lung Samples')
sample_normal = random.sample(train_images_normal,6)
f,ax = plt.subplots(2,3,Figsize=(15,9))
for i in range(0, 6):
 im = sp.imread(r'data1/train/NORMAL/' + sample_normal[i])
 print(im.shape)
 ax[i // 3, i % 3].imshow(im, cmap=plt.cm.gray)
 ax[i // 3, i % 3].axis('on')
f.suptitle('Normal Lungs')
plt.show()
print('Pneumonia Lung Samples')
sample_normal1 = random.sample(train_images_xray,6)
f,ax = plt.subplots(2,3,Figsize=(15,9))
for i in range(0, 6):
 im = sp.imread(r'data1/train/PNEUMONIA/' + sample_normal1[i])
 print(im.shape)
 ax[i // 3, i % 3].imshow(im, cmap=plt.cm.gray)
 ax[i // 3, i % 3].axis('on')
f.suptitle('Pneumonia Lungs')
plt.show()