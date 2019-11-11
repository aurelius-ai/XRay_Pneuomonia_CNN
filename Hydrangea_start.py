from os import environ, chdir
import scipy.ndimage as sp
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import random
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
chdir(r'C:\skr\hydrangea')
train_images_hortensia = os.listdir(r'data2\train\hortensia')
train_images_nature=os.listdir(r'data2\train\nature')
test_images_hortensia = os.listdir(r'data2\eval\hortensia')
test_images_nature=os.listdir(r'data2\eval\nature')
valid_images_hortensia = os.listdir(r'data2\valid\hortensia')
valid_images_nature=os.listdir(r'data2\valid\nature')
47
print('Size of Hortensia Dataset')
print('No. of images in Train data-set=',
len(train_images_hortensia)+len(train_images_nature))
print('No. of images in Test data-set=',
len(test_images_hortensia)+len(test_images_nature))
print('No. of images in Valid data-set=',
len(valid_images_hortensia)+len(valid_images_nature))
print('Hortensia Samples')
sample_normal = random.sample(train_images_hortensia,6)
f,ax = plt.subplots(2,3,Figsize=(15,9))
for i in range(0, 6):
 im = sp.imread(r'data2/train/hortensia/' + sample_normal[i])
 print(im.shape)
 ax[i // 3, i % 3].imshow(im)
 ax[i // 3, i % 3].axis('on')
f.suptitle('Hortensia')
plt.show()
print('No Hortensia Samples')
sample_normal1 = random.sample(train_images_nature,6)
f,ax = plt.subplots(2,3,Figsize=(15,9))
for i in range(0, 6):
 im = sp.imread(r'data2/train/nature/' + sample_normal1[i])
 print(im.shape)
 ax[i // 3, i % 3].imshow(im)
 ax[i // 3, i % 3].axis('on')
f.suptitle('No Hortensia')
plt.show()
