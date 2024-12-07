import os
from PIL import Image

def clean_image_directory(input_dir):
    """
    Scans a directory to identify and delete truncated or non-image files.
    
    Parameters:
    - input_dir (str): Path to the directory containing images.
    
    Returns:
    - None
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Attempt to open the image to validate it
                with Image.open(file_path) as img:
                    img.verify()  # Verify image integrity
            except Exception as e:
                # If invalid or truncated, delete the file
                os.remove(file_path)
                print(f"Deleted invalid file: {file_path}")

if __name__ == "__main__":
    # Path for your dataset
    input_directory = "./finalData/"  # Adjust the path to your dataset
    
    # Clean the directory
    clean_image_directory(input_directory)
    print("Invalid files deleted successfully.")
