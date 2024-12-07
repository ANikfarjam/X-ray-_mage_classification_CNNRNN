import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical


# Paths
train_frac_hand = 'Data/Data/Hand/train/fractured'
test_frac_hand = 'Data/Data/Hand/test/fractured'
valid_frac_hand = 'Data/Data/Hand/valid/fractured'

train_nfrac_hand = 'Data/Data/Hand/train/nonfractured'
test_nfrac_hand = 'Data/Data/Hand/test/nonfractured'
valid_nfrac_hand = 'Data/Data/Hand/valid/nonfractured'


# Prepare the Data
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_data = train_datagen.flow_from_directory(
    './Data/FractureData/training',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_data = test_datagen.flow_from_directory(
    './Data/FractureData/testing',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_data = validation_datagen.flow_from_directory(
    './Data/FractureData/validation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)


# Model
model = tf.keras.Sequential()
model.add(Conv2D(32,(3, 3), activation = 'relu', input_shape = (128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
# Converts 2D feature maps into a 1D vector to feed into fully connected layers
model.add(Dense(64, activation = 'relu'))
# Adds 64 neurons uses the relu function 
model.add(Dropout(0.5)) 
# Regularization to prevent overfitting, randomly disables 50% of neurons to prevent overfitting
model.add(Dense(10, activation = 'softmax')) 
# Output Layer adds 10 neurons


# Model Compile & Train

model.fit(train_data, train_data, epochs = 10, validation_data = validation_data)
results = model.evaluate(test_data, test_data, batch_size = 128)
print('train loss, train acc', results)

# Fractured 
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(test_data, test_data, epochs=10, validation_data=validation_data)
test_loss, test_accuracy = model.evaluate(test_data, test_data, batch_size = 128)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


