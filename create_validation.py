import os
import os.path as path
import shutil
import random
from tqdm import tqdm
print('creating validation data using %20 od test data')
#move to '/content/drive/My Drive/CS171Project/Data/avlidation'

source_dir1 = './Data/training/fractured'
source_dir2 = './Data/training/not_fractured'
dest1 = './Data/validation/fractured'
dest2 = './Data/validation/not_fractured'

"""
Make a validation dataset. Go through the BoneFractureDataset. Get 4% of the data from the non-fractured and fractured folders.
"""

#valid_files = [path.join(root, f) for root, dirs, files in os.walk(dir1) for f in files]

def create_validation_set(source, dest, desc):
  files = [os.path.join(root, f) for root, _, files in os.walk(source) for f in files]
  num_files = int(len(files)*0.02) #using only 2% of files
  files_to_move = random.sample(files, num_files)
  for file_path in tqdm(files_to_move, total=len(files_to_move), desc= desc, unit='file'):
    dest_file = os.path.join(dest, os.path.basename(file_path))
    shutil.move(file_path, dest_file)

#move files
create_validation_set(source_dir1, dest1, 'Moving fractured bones files')
create_validation_set(source_dir2, dest2, 'Moving fractured not bones files')