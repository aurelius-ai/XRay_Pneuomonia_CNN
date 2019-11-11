import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from numpy import set_printoptions
import matplotlib.pyplot as plt
import random
import scipy.ndimage as sp
from skimage import img_as_ubyte
from keras.preprocessing import image
set_printoptions(precision=4, suppress= True)
# Initial Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir(r'C:\skr\hydrangea')
# Loading Model
my_model = load_model(filepath=r'hydrangea_cnn_0.92.h5')
'''
print(my_model.summary())
'''
test_images_hortensia = os.listdir(r'data2\eval\hortensia')
test_images_nature = os.listdir(r'data2\eval\nature')
eval_idg = ImageDataGenerator(rescale=1./255)
eval_g = eval_idg.flow_from_directory(directory=r'data2\eval', target_size=(100,
100),
 class_mode='binary', batch_size=16,
shuffle=False)
(eval_loss, eval_acc) = my_model.evaluate_generator(generator=eval_g, steps=1)
print('the evaluation accuracy is {:4.2f}'.format(eval_acc))
f, ax = plt.subplots(2, 5, Figsize=(15, 9))
for i in range(0, 10):
 name = test_images_hortensia[i]
 im=sp.imread(r'C:/skr/hydrangea/data2/eval/hortensia/' + name)
 ax[i//5, i % 5].imshow(im, cmap=plt.cm.gray)
 ax[i//5, i % 5].axis('on')
 x = image.load_img(r'C:/skr/hydrangea/data2/eval/hortensia/' + name,
target_size=(100, 100))
 x = image.img_to_array(x)
 x = x / 255
 pred = my_model.predict(x.reshape((1, 100, 100, 3)))
 if pred < 0.5:
 ax[i//5, i % 5].set_title('hortensia')
 else:
 ax[i//5, i % 5].set_title('No hortensia')

f.suptitle('Hortensia Prediction by CNN')
plt.show()
f, ax = plt.subplots(2, 5, Figsize=(15, 9))
for i in range(0, 10):
 name = test_images_nature[i]
 im = sp.imread(r'C:/skr/hydrangea/data2/eval/nature/' + name)
 ax[i//5, i % 5].imshow(im, cmap=plt.cm.gray)
 ax[i//5, i % 5].axis('on')
 x = image.load_img(r'C:/skr/hydrangea/data2/eval/nature/' + name,
target_size=(100, 100))
 x = image.img_to_array(x)
 x = x / 255
 pred = my_model.predict(x.reshape((1, 100, 100, 3)))
50
 if pred < 0.5:
 ax[i//5, i % 5].set_title('hortensia')
 else:
 ax[i//5, i % 5].set_title('No hortensia')
f.suptitle('Hortensia Prediction by CNN')
plt.show()