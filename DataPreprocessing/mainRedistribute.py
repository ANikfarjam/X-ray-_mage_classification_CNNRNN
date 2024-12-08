import os
import os.path as path
from tqdm import tqdm
import numpy as np
import random
import shutil

# Positive means fractured
# Negative means unfractured

"""
Wrist
"""

source_dir1 = 'Data/Wrist/Wrist_Train_Valid/negative'
source_dir2 = 'Data/Wrist/Wrist_Train_Valid/positive'

dest1 = 'Data/Wrist/Wrist_Valid/negative'
dest2 = 'Data/Wrist/Wrist_Valid/positive'



def create_validation_set(source, dest, desc):
    # Check if dest file is full or not
    dest_files = [os.path.join(root, f) for root, _, files in os.walk(dest) for f in files]
    if len(dest_files) > 0:
      return
    else:
      files = [os.path.join(root, f) for root, _, files in os.walk(source) for f in files]
      num_files = int(len(files) * 0.2) 
      files_to_move = random.sample(files, num_files)
      for file_path in tqdm(files_to_move, total=len(files_to_move), desc = desc, unit='file'):
          dest_file = os.path.join(dest, os.path.basename(file_path))
          shutil.move(file_path, dest_file)
    

# Move Wrist Files
#create_validation_set(source_dir1, dest1, 'Moving not fractured bones into negative valid file: ')
#create_validation_set(source_dir2, dest2, 'Moving fractured bones into positive valid file: ')

"""
Shoulder
"""

shoulder_neg_dir = 'Data/ShouldersNew/Shoulder_Train_Valid/negative'
shoulder_pos_dir = 'Data/ShouldersNew/Shoulder_Train_Valid/positive'

shoulder_valid_neg = 'Data/ShouldersNew/Shoulder_Valid/negative'
shoulder_valid_pos = 'Data/ShouldersNew/Shoulder_Valid/positive'

#Move Shoulder Files
create_validation_set(shoulder_neg_dir, shoulder_valid_neg, 'Moving not fractured bones into negative valid file: ')
create_validation_set(shoulder_pos_dir, shoulder_valid_pos, 'Moving fractured bones into positive valid file: ')

shoulders_files = [os.path.join(root, f) for root, _, files in os.walk(shoulder_valid_neg) for f in files]
print(len(shoulders_files)) # 153 Valid Neg

"""
Humerus
"""

humerus_neg_dir = 'Data/Humerus/Humerus_Train_Valid/negative'
humerus_pos_dir = 'Data/Humerus/Humerus_Train_Valid/positive'

humerus_valid_neg = 'Data/Humerus/Humerus_Valid/negative'
humerus_valid_pos = 'Data/Humerus/Humerus_Valid/positive'

#Move Shoulder Files
create_validation_set(humerus_neg_dir, humerus_valid_neg, 'Moving not fractured bones into negative valid file: ')
create_validation_set(humerus_pos_dir, humerus_valid_pos, 'Moving fractured bones into positive valid file: ')

humerus_files = [os.path.join(root, f) for root, _, files in os.walk(humerus_valid_neg) for f in files]
print(len(humerus_files)) 

"""
Hand Part 2
"""

hand2_neg_dir = 'Data/Hand_Part_2/Hand_Train_Valid/negative'
hand2_pos_dir = 'Data/Hand_Part_2/Hand_Train_Valid/positive'

hand2_valid_neg = 'Data/Hand_Part_2/Hand_Valid/negative'
hand2_valid_pos = 'Data/Hand_Part_2/Hand_Valid/positive'

#Move Hand Files
create_validation_set(hand2_neg_dir, hand2_valid_neg, 'Moving not fractured bones into negative valid file: ')
create_validation_set(hand2_pos_dir, hand2_valid_pos, 'Moving fractured bones into positive valid file: ')

#Check to see if it's the right size.
test_files = [os.path.join(root, f) for root, _, files in os.walk(hand2_valid_neg) for f in files]
print(len(test_files))

"""
Forearm
"""

forearm_neg_dir = 'Data/Forearm/Forearm_Train_Valid/negative'
forearm_pos_dir = 'Data/Forearm/Forearm_Train_Valid/positive'

forearm_valid_neg = 'Data/Forearm/Forearm_Valid/negative'
forearm_valid_pos = 'Data/Forearm/Forearm_Valid/positive'

#Move Forearm Files
create_validation_set(forearm_neg_dir, forearm_valid_neg, 'Moving not fractured bones into negative valid file: ')
create_validation_set(forearm_pos_dir, forearm_valid_pos, 'Moving fractured bones into positive valid file: ')
# 122 Valid Neg

"""
Finger
"""
finger_neg_dir = 'Data/Finger/Finger_Train_Valid/negative'
finger_pos_dir = 'Data/Finger/Finger_Train_Valid/positive'

finger_valid_neg = 'Data/Finger/Finger_Valid/negative'
finger_valid_pos = 'Data/Finger/Finger_Valid/positive'

create_validation_set(finger_neg_dir, finger_valid_neg, 'Moving not fractured bones into negative valid file: ')
create_validation_set(finger_pos_dir, finger_valid_pos, 'Moving fractured bones into positive valid file: ')
# 145
"""
Elbow
"""
elbow_neg_dir = 'Data/Elbow/Elbow_Train_Valid/negative'
elbow_pos_dir = 'Data/Elbow/Elbow_Train_Valid/positive'

elbow_valid_neg = 'Data/Elbow/Elbow_Valid/negative'
elbow_valid_pos = 'Data/Elbow/Elbow_Valid/positive'

create_validation_set(elbow_neg_dir, elbow_valid_neg, 'Moving not fractured bones into negative valid file: ')
create_validation_set(elbow_pos_dir, elbow_valid_pos, 'Moving fractured bones into positive valid file: ')

#160