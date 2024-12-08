import os
import os.path as path
from tqdm import tqdm
import numpy as np
import random
import shutil
from mainRedistribute import create_validation_set


"""
Humerus
"""

humerus_neg_dir = 'Data/Humerus/Humerus_Train_Valid/negative'
humerus_pos_dir = 'Data/Humerus/Humerus_Train_Valid/positive'

humerus_valid_neg = 'Data/Humerus/Humerus_Valid/negative'
humerus_valid_pos = 'Data/Humerus/Humerus_Valid/positive'

# Move Humerus Files
create_validation_set(humerus_neg_dir, humerus_valid_neg, 'Moving not fractured bones into negative valid file: ')
create_validation_set(humerus_pos_dir, humerus_valid_pos, 'Moving fractured bones into positive valid file: ')

