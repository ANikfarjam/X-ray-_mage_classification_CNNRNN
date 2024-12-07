import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
import pandas as pd
import os
import tqdm
import cv2
from PIL import Image

# Check for GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Repair truncated images
def repairImage(img_path):
    img = cv2.imread(img_path)
    try:
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[np.where(img == 0)] = 255  # Create a mask for the truncated area
        inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        save_path = os.path.splitext(img_path)[0] + '_repaired.jpg'
        success = cv2.imwrite(save_path, inpainted_img)
        os.remove(img_path)
        if success:
            print(f"Repaired image saved to: {save_path}")
            return save_path
        else:
            print(f"Failed to save the repaired image to: {save_path}")
    except FileNotFoundError as fnf_error:
        print(f"File error: {fnf_error}")
    except cv2.error as cv_error:
        print(f"OpenCV error: {cv_error}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Check for truncation and repair if necessary
def check_images(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify if the image is valid
            print(f"{image_path} is good!")
    except (IOError, SyntaxError) as e:
        print(f"Image verification failed for {image_path}: {e}")
        repairImage(image_path)

# Preprocessing function for Fourier Transform
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
    if len(fft_image.shape) < 3 or fft_image.shape[-1] != 3:
        fft_image = tf.stack([fft_image] * 3, axis=-1)

    # Normalize to [0, 1]
    fft_image = fft_image / tf.reduce_max(fft_image)

    return fft_image

# Function to preprocess dataset
def preprocess_dataset(data_path):
    image_list = []
    file_paths = []
    fetched_images = [os.path.join(root, f) for root, _, files in os.walk(data_path) for f in files if f.endswith('.png')]

    for img_path in tqdm.tqdm(fetched_images, desc=f"Processing {data_path}", unit='file'):
        check_images(img_path)
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        fft_array = img_fft(img_array)
        image_list.append(fft_array)
        file_paths.append(img_path)

    return np.array(image_list), file_paths

# Pseudo-labeling function
def pseudo_label_data(model, dataset, file_paths, category):
    predictions = model.predict(dataset)
    predicted_labels = np.argmax(predictions, axis=1)  # Assuming classification model

    df = pd.DataFrame({
        "file_path": file_paths,
        "predicted_label": predicted_labels
    })

    output_file = f"pseudo_labels_{category}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved pseudo-labels for {category} to {output_file}")

# Load the base model
handBaseModel = tf.keras.models.load_model('hand_best_base_model.h5')

# Paths for each dataset
datasets = {
    "Wrist_frac_test": "Data/Data/Wrist/test/fractured",
    "Wrist_frac_train": "Data/Data/Wrist/train/fractured",
    "Wrist_frac_valid": "Data/Data/Wrist/valid/fractured",
    "Wrist_nfrac_test": "Data/Data/Wrist/test/nonfractured",
    "Wrist_nfrac_train": "Data/Data/Wrist/train/nonfractured",
    "Wrist_nfrac_valid": "Data/Data/Wrist/valid/nonfractured"
}

# Process and pseudo-label all datasets
for category, path in datasets.items():
    print(f"Processing dataset: {category}")
    dataset, file_paths = preprocess_dataset(path)
    pseudo_label_data(handBaseModel, dataset, file_paths, category)
