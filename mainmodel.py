import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
import tqdm
import pywt  # Wavelet transform
from pprint import pprint

# Logging setup
class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "w")
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

sys.stdout = Tee("main_model_logs.txt")

print("Checking available GPUs...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Number of GPUs Available: {len(gpus)}")
else:
    print("No GPU detected. Ensure CUDA is properly installed.")

print("****************** Initialization ******************")
print("Starting the script... Logs will be saved to 'main_model_logs.txt'.")

# Define custom labels
labels = {
    'hands/fractured': [1, 0],
    'hands/non-fractured': [0, 0],
    'elbow/fractured': [1, 1],
    'elbow/non-fractured': [0, 1],
    'humerus/fractured': [1, 2],
    'humerus/non-fractured': [0, 2],
    'shoulders/fractured': [1, 3],
    'shoulders/non-fractured': [0, 3],
    'legs/fractured': [1, 4],
    'legs/non-fractured': [0, 4],
    'hip/fractured': [1, 5],
    'hip/non-fractured': [0, 5],
    'knee/fractured': [1, 7],
    'knee/non-fractured': [0, 7],
    'spine/fractured': [1, 8],
    'spine/non-fractured': [0, 8],
    'forearm/fractured': [1, 9],
    'forearm/non-fractured': [0, 9]
}
print("Custom labels initialized.")

# Preprocessing functions
def img_preprocess(image_path, method='fft'):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")
        
        image = cv2.resize(image, (128, 128))

        if method == 'fft':
            fft_image = np.fft.fft2(image)
            fft_image = np.fft.fftshift(fft_image)
            fft_magnitude = np.abs(fft_image)
            fft_magnitude = np.log(1 + fft_magnitude)
            processed_image = fft_magnitude / np.max(fft_magnitude)
        elif method == 'wavelet':
            coeffs2 = pywt.dwt2(image, 'haar')
            cA, (cH, cV, cD) = coeffs2
            processed_image = np.log(1 + np.abs(cA)) / np.max(np.abs(cA))
        else:
            raise ValueError("Invalid preprocessing method. Choose 'fft' or 'wavelet'.")

        processed_image = np.expand_dims(processed_image, axis=-1)
        processed_image = np.concatenate([processed_image] * 3, axis=-1)
        return processed_image
    except Exception as e:
        print(f"[ERROR] Preprocessing failed for {image_path}: {e}")
        return None

def preprocess_images(image_dir, desc, method='fft'):
    processed_images = []
    labels_list = []
    failed_files = []

    print(f"[INFO] Preprocessing images in directory: {image_dir}")
    data_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(image_dir)
        for file in files if file.lower().endswith(('.jpg', '.png'))
    ]

    for img_path in tqdm.tqdm(data_paths, total=len(data_paths), desc=desc, unit='file'):
        try:
            with Image.open(img_path) as img:
                img.verify()

            processed_image = img_preprocess(img_path, method)
            if processed_image is None:
                failed_files.append(img_path)
                continue

            processed_images.append(processed_image)
            body_part = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            condition = os.path.basename(os.path.dirname(img_path))
            label_key = f"{body_part}/{condition}"

            if label_key in labels:
                # Use the second value of the label (integer encoding)
                labels_list.append(labels[label_key][1])
            else:
                print(f"[WARNING] Label '{label_key}' not found.")
                failed_files.append(img_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")
            failed_files.append(img_path)

    if failed_files:
        with open('failed_files.log', 'w') as log_file:
            log_file.write("\n".join(failed_files))
        print(f"[INFO] Skipped {len(failed_files)} invalid images. See 'failed_files.log'.")

    return np.array(processed_images), np.array(labels_list)

# Data preprocessing
train_images, train_labels = preprocess_images('./finalData/train', 'Preprocessing Training Images!', method='wavelet')
test_images, test_labels = preprocess_images('./finalData/test', 'Preprocessing Testing Images!', method='wavelet')
val_images, val_labels = preprocess_images('./finalData/validation', 'Preprocessing Validation Images!', method='wavelet')

# Build and evaluate the model
def build_model(hp):
    model = models.Sequential()
    for i in range(hp.Int('conv_blocks', 1, 3)):
        model.add(layers.Conv2D(filters=hp.Int(f'filters_{i}', 32, 128, step=32),
                                kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Reshape((1, -1)))

    model.add(layers.Bidirectional(layers.LSTM(units=hp.Int('lstm_units', 32, 128, step=32), return_sequences=False)))
    model.add(layers.Dense(len(labels), activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def test_model(model, test_images, test_labels):
    y_pred_probs = model.predict(test_images)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(test_labels, y_pred, output_dict=True)
    cm = confusion_matrix(test_labels, y_pred)

    metrics_df = pd.DataFrame(report).transpose()
    cm_df = pd.DataFrame(cm)

    metrics_df.to_csv('classification_report.csv', index=False)
    cm_df.to_csv('confusion_matrix.csv', index=False)

    print("Saved performance metrics and confusion matrix.")

if __name__ == '__main__':
    tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=10, directory='main_nas_dir')
    tuner.search(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
    best_model = tuner.get_best_models(1)[0]
    test_model(best_model, test_images, test_labels)
    best_model.save('Main_model_saved.h5')

