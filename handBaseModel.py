# Author: ANikfarjam
import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

 
# Check for GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if __name__=='__main__':
    #########Set UP NAS###########
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
    ###############preparing data##########
    # Preprocessing function for Fourier Transform
    def img_fft(image):
        # Resize to target dimensions
        image = tf.image.resize(image, (128, 128))

        # Apply FFT
        fft_image = tf.signal.fft2d(tf.cast(image, tf.complex64))  # FFT returns complex numbers
        fft_image = tf.abs(fft_image)  # Use the magnitude

        # Normalize the FFT output
        fft_image = tf.math.log(1 + fft_image)  # Log scaling for visualization

        # Ensure we have 3 channels
        if len(fft_image.shape) < 3 or fft_image.shape[-1] != 3:
            fft_image = tf.stack([fft_image] * 3, axis=-1)

        return fft_image


    batch_size = 32
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalize pixel values
        rotation_range=90,  # Augmentation
        preprocessing_function=img_fft  # Apply our custom FFT preprocessing
    )

    # Ensure `target_size` matches your FFT resizing (128x128)
    train_data = data_gen.flow_from_directory(
        './Data/Data/Hand/train',
        target_size=(128, 128),  # Matches resizing in `img_fft`
        color_mode='rgb',       # Ensures 3 channels are expected
        class_mode='binary',
        batch_size=batch_size
    )
    validation_data = data_gen.flow_from_directory(
        './Data/Data/Hand/val',
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='binary',
        batch_size=batch_size
    )
    test_data = data_gen.flow_from_directory(
        './Data/Data/Hand/test',
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='binary',
        batch_size=batch_size
    )
    ########## NAS Search ##########
    # NAS Search and Training
    tuner.search(train_data, epochs=10, validation_data=validation_data)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save('hand_best_base_model.h5')
    # Evaluate the best model
    loss, accuracy = best_model.evaluate(validation_data)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


 