import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import os
import os.path as path
from tqdm import tqdm
from PIL import Image
import scipy as sp
import sklearn
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image, ImageFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras_tuner import RandomSearch


# CNN
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(Conv2D(filters = 32, kernel_size= (3, 3), activation = 'relu', input_shape = (128, 128, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


data_gen = ImageDataGenerator(1.0/255)
batch_size = 32
# Ensure `target_size` matches your FFT resizing (128x128)
"""
Training, Testing, Validation
"""
# Train Data 
train_data = data_gen.flow_from_directory(
        # Training Path
        './Data/elbow/train',
        target_size=(128, 128),  
        color_mode='grayscale',       
        class_mode='binary',
        batch_size=batch_size
)

validation_data = data_gen.flow_from_directory(
        # Validation Path
        #"./Data/legs/valid"
        './Data/elbow/valid',
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode='binary',
        batch_size=batch_size
)

test_data = data_gen.flow_from_directory(
        # Test Path
        './Data/elbow/test',
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode='binary',
        batch_size=batch_size
)

tuner.search(train_data, epochs = 10, validation_data = validation_data)

# Get the best model
CNN_best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
loss, accuracy = CNN_best_model.evaluate(validation_data)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
