import os
from PIL import Image

def repair_image(image_path):
    """
    Deletes truncated or corrupted images.
    """
    try:
        # Attempt to load the image
        with Image.open(image_path) as img:
            img.verify()  # Verify image integrity
        print(f"[INFO] {image_path} is valid.")
        return image_path  # Return the valid image path
    except (IOError, SyntaxError) as e:
        print(f"[WARNING] Deleting corrupted image: {image_path}, Error: {e}")
        try:
            os.remove(image_path)  # Delete the corrupted file
            print(f"[SUCCESS] Deleted {image_path}.")
        except Exception as delete_error:
            print(f"[ERROR] Failed to delete {image_path}: {delete_error}")
        return None  # Return None to indicate the file was deleted

def repair_directory(image_dir):
    """
    Iterates through a directory to validate and repair/delete images.
    """
    print(f"[INFO] Repairing images in directory: {image_dir}")
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(root, file)
                repair_image(img_path)

if __name__ == "__main__":
    # Path to the directory containing images
    image_directory = "./finalData/"  # Update this path as needed
    repair_directory(image_directory)
    print("[INFO] Repair process completed.")
