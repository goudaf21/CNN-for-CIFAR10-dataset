import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

## Load is Cifar 10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

## Print ouf shapes of training and testing sets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

## Normalize x train and x test images
X_train = X_train.astype('float') / 255
X_test = X_test.astype('float') / 255

## Create one hot encoding vectors for y train and y test
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

## Define the model
model = Sequential()

## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, same padding and input shape
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',
                    kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
print('Conv2D added')

## Add a batch normalization layer
model.add(BatchNormalization())
print('BatchNormalization added')

## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                    kernel_initializer='he_uniform', padding='same'))
print('Conv2D added')

## Add a batch normalization layer
model.add(BatchNormalization())
print('BatchNormalization added')

## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size=(2, 2)))
print("MaxPooling2D added")

## Add dropout layer of 0.2
model.add(Dropout(0.2))
print("Dropout added")

## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',
                    kernel_initializer='he_uniform', padding='same'))
print('Conv2D added')

## Add a batch normalization layer
model.add(BatchNormalization())
print('BatchNormalization added')

## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',
                    kernel_initializer='he_uniform', padding='same'))
print('Conv2D added')

## Add a batch normalization layer
model.add(BatchNormalization())
print('BatchNormalization added')

## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size=(2, 2)))
print("MaxPooling2D added")

## Add dropout layer of 0.2
model.add(Dropout(0.2))
print("Dropout added")

## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',
                    kernel_initializer='he_uniform', padding='same'))
print('Conv2D added')

## Add a batch normalization layer
model.add(BatchNormalization())
print('BatchNormalization added')

## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',
                    kernel_initializer='he_uniform', padding='same'))
print('Conv2D added')

## Add a batch normalization layer
model.add(BatchNormalization())
print('BatchNormalization added')

## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size=(2, 2)))
print("MaxPooling2D added")

## Add dropout layer of 0.2
model.add(Dropout(0.2))
print("Dropout added")

## Flatten the resulting data
model.add(Flatten())
print("Flatten added")

## Add a dense layer with 128 nodes, relu activation and he uniform kernel initializer
model.add(Dense(units=128, activation='relu', kernel_initializer='he_uniform'))
print("Dense added")

## Add a batch normalization layer
model.add(BatchNormalization())
print('BatchNormalization added')

## Add dropout layer of 0.2
model.add(Dropout(0.2))
print("Dropout added")

## Add a dense softmax layer
model.add(Dense(units=10, activation='softmax'))
print("Softmax added")

## Set up early stop training with a patience of 3
early_stop = EarlyStopping(patience=3)
print("Early stop object created")

## Compile the model with adam optimizer, categorical cross entropy and accuracy metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled")

# Image Data Generator , we are shifting image accross width and height of 0.1 also we are flipping the image horizantally and rotating the images by 20 degrees
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, rotation_range=20)
datagen.fit(X_train)
print("Image Data Generator object created")

## Take data and label arrays to generate a batch of augmented data, default parameters are fine.
iterator = datagen.flow(X_train, y_train, shuffle=False)
batch_images, batch_labels = next(iterator)

## Define the number of steps to take per epoch as training examples over 64
steps_per_epoch = len(X_train) / 64
epochs=200

## Fit the model with the generated data, 200 epochs, steps per epoch and validation data defined.
fitted_model = model.fit(iterator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=(X_test, y_test))

#Show graph of loss over time for training data
num = range(1, epochs+1)
plt.plot(num, fitted_model.history['loss'], '-b', label='loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over time for training data")
plt.show()

#Show graph of accuracy over time for training data
num = range(1, epochs+1)
plt.plot(num, fitted_model.history['accuracy'], '-b', label='accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy over time for training data")
plt.show()

print("Accuracy :", model.evaluate(X_test, y_test)[1])
