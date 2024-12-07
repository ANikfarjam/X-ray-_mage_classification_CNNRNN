#Data Augmentation for validation set
#even though NAS does this but since our test and train data is already augmented we need to do the same for validation
from PIL import Image
import os
from tqdm import tqdm
from multiprocessing import Pool
# image augmentaion
def augment_image(image_path):
    img = Image.open(image_path)
    #resized image
    resized_img = img.resize((200,200))
    #rotated image
    rotated_img1 = img.rotate(45) #45 degree rotation
    rotated_img2 = img.rotate(90) #90 degree rotation
    #fliped_image
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Save the augmented images
    resized_img.save(image_path + "_resized.jpg")
    rotated_img1.save(image_path + "_rotated1.jpg")
    rotated_img2.save(image_path + "_rotated2.jpg")
    flipped_img.save(image_path + "_flipped.jpg")
    #now that created our validation files lets augment them
validation_dir = './Data/FractureData/validation'
files_to_augment = [os.path.join(root, f) for root, _, files in os.walk(validation_dir) for f in files]
with Pool() as pool:
  list(tqdm(pool.imap(augment_image, files_to_augment), total=len(files_to_augment), desc="Augmenting data", unit="file"))