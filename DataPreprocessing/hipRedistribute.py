# Hip All Fracture

# Hip Fracture Or NonFracture

# Go through each test, train and valid and start determine if it's fractured or non fractured

import os
import os.path as path
from tqdm import tqdm
import shutil


"""
Make a fractured and nonfractured directory for reach subfolder test, train and valid 
"""

# Paths
hipTrain_images = 'Data/Data/Hip/HipSubFolder/HipF&NF/train/images'

hipTest_images = 'Data/Data/Hip/HipSubFolder/HipF&NF/test/images'

hipValid_images = 'Data/Data/Hip/HipSubFolder/HipF&NF/valid/images'

# Create New Dirs / Destination
hipTrain_frac = '../Data/Data/Hip/HipSubFolder/HipF&NF/train/fractured'
hipTrain_nfrac = '../Data/Data/Hip/HipSubFolder/HipF&NF/train/nonfractured'

hipTest_frac = '../Data/Data/Hip/HipSubFolder/HipF&NF/test/fractured'
hipTest_nfrac = '../Data/Data/Hip/HipSubFolder/HipF&NF/test/nonfractured'

hipValid_frac = '../Data/Data/Hip/HipSubFolder/HipF&NF/valid/fractured'
hipValid_nfrac = '../Data/Data/Hip/HipSubFolder/HipF&NF/valid/nonfractured'

def hipRedistribute(source, frac_dest, non_dest):
    # All Images
    """
    If image file ends in g, then it's normal

    If image file ends in f then, it's fractured
    """
    
    for file_name in os.listdir(source):
        file_path = os.path.join(source, file_name)
        
        # Normal
        if file_name.split('_')[0][-1] == 'g':
            shutil.move(file_path, os.path.join(non_dest, file_name))
        
        # Fractured
        if file_name.split('_')[0][-1] == 'f':
            shutil.move(file_path, os.path.join(frac_dest, file_name))



# Train

#hipRedistribute(hipTrain_images, hipTrain_frac, hipTrain_nfrac)

# Test
hipRedistribute(hipTest_images, hipTest_frac, hipTest_nfrac)

# Valid
#hipRedistribute(hipValid_images, hipValid_frac, hipValid_nfrac)


"""
Find the count of all files



# Train
HipTrain_frac_files = [os.path.join(root, f) for root, _, files in os.walk(hipTrain_frac) for f in files]

HipTrain_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(hipTrain_nfrac) for f in files]

# Test
HipTest_frac_files = [os.path.join(root, f) for root, _, files in os.walk(hipTest_frac) for f in files]

HipTest_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(hipTest_nfrac) for f in files]

# Valid
HipValid_frac_files = [os.path.join(root, f) for root, _, files in os.walk(hipValid_frac) for f in files]

HipValid_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(hipValid_nfrac) for f in files]
"""