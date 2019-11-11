from os import environ, chdir
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers, callbacks
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D
import matplotlib.pyplot as plt
44
from skimage import img_as_ubyte
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
chdir(r'C:\skr\xray')
# Setting Image Generators
train_idg = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True)
train_g = train_idg.flow_from_directory(directory=r'data1\train', target_size=(150,
150),
 class_mode='binary', batch_size=32)
valid_idg = ImageDataGenerator(rescale=1./255)
valid_g = valid_idg.flow_from_directory(directory=r'data1\val', target_size=(150,
150),
 class_mode='binary', batch_size=32)
# CNN Architecture
my_model = Sequential()
my_model.add(Conv2D(64,(3,3),input_shape=(150,150,3),activation='relu'))
my_model.add(Conv2D(64,(3,3),activation='relu'))
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Dropout(0.2))
my_model.add(Conv2D(32,(3,3),activation='relu'))
my_model.add(Conv2D(32,(3,3),activation='relu'))
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Dropout(0.2))
my_model.add(Conv2D(16,(3,3),activation='relu'))
my_model.add(Conv2D(16,(3,3),activation='relu'))
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Dropout(0.2))
my_model.add(Flatten())
my_model.add(Dense(units=128,activation='relu'))
my_model.add(Dropout(0.2))
my_model.add(Dense(units=1,activation='sigmoid'))
print(my_model.summary())
# Model Loss Function and Optimiser
my_model.compile(optimizer='adam', loss='binary_crossentropy',
metrics=['accuracy'])
# Callbacks
check_p = callbacks.ModelCheckpoint(filepath='xray9_cnn_{val_acc:.2f}.h5',
monitor='val_acc',
 verbose=1, save_best_only=True,
save_weights_only=False)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2,
cooldown=2, min_lr=0.00001, verbose=1)
call_l = [check_p, reduce_lr]
# Training Options
fit = my_model.fit_generator(generator=train_g, steps_per_epoch=163, epochs=13,
verbose=1, callbacks=call_l,
 validation_data=valid_g, validation_steps=19)
print(fit.history.keys())
# summarize history for accuracy
plt.plot(fit.history['acc'])
plt.plot(fit.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
45
plt.show()
# summarize history for loss
plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# Saving Model
my_model.save(filepath=r'xray9_cnn.h5', overwrite=True)
