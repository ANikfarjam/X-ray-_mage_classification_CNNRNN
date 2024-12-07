"""
My original CNN was in a journal page.
This is just that but instead its in a python page.
"""

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
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def img_fft(image):
    # Resize to target dimensions
    image = tf.image.resize(image, (128, 128))

    # Apply FFT
    fft_image = tf.signal.fft2d(tf.cast(image, tf.complex64))
    fft_image = tf.abs(fft_image)
    fft_image = tf.math.log(1 + fft_image)

    # Normalize the FFT output
    fft_image = tf.image.resize(fft_image, (128, 128))

    # Ensure we have 3 channels
    #if len(fft_image.shape) < 3 or fft_image.shape[-1] != 3:
    #   fft_image = tf.stack([fft_image] * 3, axis=-1)

    # Normalize to [0, 1]
    fft_image = fft_image / tf.reduce_max(fft_image)

    return fft_image



batch_size = 32
data_gen = ImageDataGenerator(
            rescale= 1.0 / 255,  # Normalize pixel values
            rotation_range = 90,  # Augmentation
            preprocessing_function = img_fft  # Apply our custom FFT preprocessing
    )

# Ensure `target_size` matches your FFT resizing (128x128)
train_data = data_gen.flow_from_directory(
        # Train Path
        target_size=(128, 128),  # Matches resizing in `img_fft`
        color_mode='grayscale',       # Ensures 3 channels are expected
        class_mode='binary',
        batch_size=batch_size
)

validation_data = data_gen.flow_from_directory(
        # Validation Path
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode='binary',
        batch_size=batch_size
)

test_data = data_gen.flow_from_directory(
        # Test Path
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode='binary',
        batch_size=batch_size
)


model = tf.keras.Sequential()
model.add(Conv2D(filters = 32, kernel_size= (3, 3), activation = 'relu', input_shape = (128, 128, 1)))
"""
Conv 2D Filter: 32 Layers Kernel Size: 3 by 3 Sliding Window Activation: Relu Input Shape: Height = 128, Width = 128, Channels = 1
"""
model.add(BatchNormalization())
"""
Batch Normalization improves training speed and stability for neural networks. 
"""
model.add(MaxPooling2D(pool_size = (2, 2)))
"""
Dimensionality Reduction and prevents overfitting by pooling the windows, each 2 by 2 is reduced to a maximum value.
"""
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
"""
Converts multi dimensional input data into a one-dimensional vector
"""
model.add(Dense(64, activation = 'relu'))
# Dense creates a layer of 64
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
# Dense creates a node that determines if it's fractured or not.

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(train_data, epochs = 10, validation_data = validation_data)



test_loss, test_accuracy = model.evaluate(test_data)

print("Test Accuracy: ", test_accuracy)
print("Test Loss: ", test_loss)