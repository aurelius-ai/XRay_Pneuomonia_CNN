import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from numpy import set_printoptions
import random
import matplotlib.pyplot as plt
import scipy.ndimage as sp
from keras.preprocessing import image
set_printoptions(precision=4, suppress= True)
# Initial Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir(r'C:\skr\xray')
# Loading Model
my_model = load_model(filepath=r'xray6_cnn_0.91.h5')
print(my_model.summary())
# Weights and biases
'''
print('Hydrangea last layer weights')
print(my_model.get_weights()[-2])
'''
# Evaluation
eval_idg = ImageDataGenerator(rescale=1./255)
eval_g = eval_idg.flow_from_directory(directory=r'data1\test', target_size=(150,
150),
 class_mode='binary', batch_size=16,
shuffle=False)
(eval_loss, eval_acc) = my_model.evaluate_generator(generator=eval_g, steps=1)
print('the evaluation accuracy is {:4.2f}'.format(eval_acc))
# Individual Prediction
'''
pre_idg = eval_idg
pre_g = eval_g
pre = my_model.predict_generator(generator=pre_g, steps=1)
print(pre_g.filenames, '\n')
print(pre_g.class_indices, '\n')
print(pre[0:8]<0.5, '\n')
46
print(pre[8:16]>0.5, '\n')
'''
test_images_normal = os.listdir(r'data1\test\NORMAL')
test_images_xray = os.listdir(r'data1\test\PNEUMONIA')
f, ax = plt.subplots(2, 4, Figsize=(15, 9))
for i in range(0, 8):
 name = test_images_normal[i]
 im = sp.imread(r'C:/skr/xray/data1/test/NORMAL/' + name)
 ax[i // 4, i % 4].imshow(im, cmap=plt.cm.gray)
 ax[i // 4, i % 4].axis('on')
 x = image.load_img(r'C:/skr/xray/data1/test/NORMAL/' + name, target_size=(150,
150))
 x = image.img_to_array(x)
 x = x / 255
 pred = my_model.predict(x.reshape((1, 150, 150, 3)))
 if pred < 0.5:
 ax[i // 4, i % 4].set_title('Normal')
 else:
 ax[i // 4, i % 4].set_title('Pneumonia')
f.suptitle('Normal Scan Prediction by CNN')
plt.show()
f, ax = plt.subplots(2, 4, Figsize=(15, 9))
for i in range(0, 8):
 name = test_images_xray[i]
 im = sp.imread(r'C:/skr/xray/data1/test/PNEUMONIA/' + name)
 ax[i // 4, i % 4].imshow(im, cmap=plt.cm.gray)
 ax[i // 4, i % 4].axis('on')
 x = image.load_img(r'C:/skr/xray/data1/test/PNEUMONIA/' + name,
target_size=(150, 150))
 x = image.img_to_array(x)
 x = x / 255
 pred = my_model.predict(x.reshape((1, 150, 150, 3)))
 if pred < 0.5:
 ax[i // 4, i % 4].set_title('Normal')
 else:
 ax[i // 4, i % 4].set_title('Pneumonia')
f.suptitle('Pneumonia Scan Prediction by CNN')
plt.show()