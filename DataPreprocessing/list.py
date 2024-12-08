import os
import os.path as path
import shutil


# Paths
hipTrain_images = 'Data/Hip/HipF&NF/train/images'
hipTest_images = 'Data/Hip/HipF&NF/test/images'
hipValid_images = 'Data/Hip/HipF&NF/valid/images'

# Destination
hipTrain_frac = 'Data/Hip/HipF&NF/train/fractured'
hipTrain_nfrac = 'Data/Hip/HipF&NF/train/nonfractured'

hipTest_frac = 'Data/Hip/HipF&NF/test/fractured'
hipTest_nfrac = 'Data/Hip/HipF&NF/test/nonfractured'

hipValid_frac = 'Data/Hip/HipF&NF/valid/fractured'
hipValid_nfrac = 'Data/Hip/HipF&NF/valid/nonfractured'


def hipRedistribute(source, frac_dest, non_dest):
    for file_name in os.listdir(source):
        file_path = os.path.join(source, file_name)
        
        # Normal
        if file_name.split('_')[0][-1] == 'g':
            shutil.move(file_path, os.path.join(non_dest, file_name))
        
        # Fractured
        if file_name.split('_')[0][-1] == 'f':
            shutil.move(file_path, os.path.join(frac_dest, file_name))


#hipRedistribute(hipTrain_images, hipTrain_frac, hipTrain_nfrac)
#hipRedistribute(hipTest_images, hipTest_frac, hipTest_nfrac)
#hipRedistribute(hipValid_images, hipValid_frac, hipValid_nfrac)






