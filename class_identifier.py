from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the same data generator as in the main script
data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=90,
    preprocessing_function=None  # No preprocessing needed for this task
)

# Path to the training directory
train_data_path = './Data/Data/Hand/train'

# Load the data and map the labels
train_data = data_gen.flow_from_directory(
    train_data_path,
    target_size=(128, 128),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False  # Disable shuffle to match labels with directory names
)

# Extract the class indices
class_indices = train_data.class_indices

# Reverse the dictionary to map 0 and 1 to class names
label_mapping = {v: k for k, v in class_indices.items()}

print("Label Definitions:")
for label, class_name in label_mapping.items():
    print(f"{label}: {class_name}")
