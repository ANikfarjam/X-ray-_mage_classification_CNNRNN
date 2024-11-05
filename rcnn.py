import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Check for GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs found: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected")

# Define the model with Hyperparameters for NAS
def build_model(hp):
    model = models.Sequential()

    # CNN Layers
    for i in range(hp.Int('conv_blocks', 1, 3)):
        model.add(layers.Conv2D(filters=hp.Int(f'filters_{i}', 32, 128, step=32),
                                kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    # RNN Layer
    model.add(layers.Reshape((1, -1)))  # Adjust this reshape based on input size
    model.add(layers.LSTM(units=hp.Int('lstm_units', 32, 128, step=32), return_sequences=False))

    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Set up the Hyperband tuner for NAS
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    directory='nas_dir',
    project_name='cnn_rnn_nas'
)

# Prepare the Data
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_data = train_datagen.flow_from_directory(
    './Data/training',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_data = test_datagen.flow_from_directory(
    './Data/testing',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_data = validation_datagen.flow_from_directory(
    './Data/validation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# NAS Search and Training
tuner.search(train_data, epochs=10, validation_data=validation_data)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
loss, accuracy = best_model.evaluate(validation_data)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")