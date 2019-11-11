from os import environ, chdir
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers, callbacks
import matplotlib.pyplot as plt
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
chdir(r'C:\skr\hydrangea')
# Setting Image Generators
train_idg = ImageDataGenerator(rescale=1./255, zoom_range=[1.0, 1.25],
width_shift_range=0.1,
 height_shift_range=0.1, fill_mode='reflect')
train_g = train_idg.flow_from_directory(directory=r'data2\train', target_size=(100,
100),
 class_mode='binary', batch_size=125,
shuffle=True)
valid_idg = ImageDataGenerator(rescale=1./255)
valid_g = valid_idg.flow_from_directory(directory=r'data2\valid', target_size=(100,
100),
 class_mode='binary', batch_size=125,
shuffle=True)
# CNN Architecture
my_model = models.Sequential()
my_model.add(layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1),
input_shape=(100, 100, 3)))
my_model.add(layers.Activation('relu'))
my_model.add(layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1),
input_shape=(100, 100, 3)))
my_model.add(layers.Activation('relu'))
my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
my_model.add(layers.Dropout(rate=0.3))
48
my_model.add(layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1)))
my_model.add(layers.Activation('relu'))
my_model.add(layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1)))
my_model.add(layers.Activation('relu'))
my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
my_model.add(layers.Dropout(rate=0.3))
my_model.add(layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1)))
my_model.add(layers.Activation('relu'))
my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
my_model.add(layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1)))
my_model.add(layers.Activation('relu'))
my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
my_model.add(layers.Flatten())
my_model.add(layers.Dropout(rate=0.3))
my_model.add(layers.Dense(units=10))
my_model.add(layers.Activation('relu'))
my_model.add(layers.Dense(units=1))
my_model.add(layers.Activation('sigmoid'))
print(my_model.summary())
# Model Loss Function and Optimiser
compile1 = my_model.compile(optimizer=optimizers.sgd(lr=0.15),
loss='binary_crossentropy', metrics=['accuracy'])
# Callbacks
check_p = callbacks.ModelCheckpoint(filepath='hydrangea1_cnn_{val_acc:.2f}.h5',
monitor='val_acc',
 verbose=1, save_best_only=True,
save_weights_only=False)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.95,
verbose=1, cooldown=2)
call_l = [check_p, reduce_lr]
# Training Options
fit = my_model.fit_generator(generator=train_g, steps_per_epoch=22, epochs=100,
verbose=1, callbacks=call_l,
 validation_data=valid_g, validation_steps=4)
print(fit.history.keys())
# summarize history for accuracy
plt.plot(fit.history['acc'])
plt.plot(fit.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
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
my_model.save(filepath=r'hydrangea1_cnn.h5', overwrite=True)