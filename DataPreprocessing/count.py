
import os
import os.path as path

hipTrain_frac = 'Data/Hip/HipF&NF/train/fractured'
hipTrain_nfrac = 'Data/Hip/HipF&NF/train/nonfractured'

hipTest_frac = 'Data/Hip/HipF&NF/test/fractured'
hipTest_nfrac = 'Data/Hip/HipF&NF/test/nonfractured'

hipValid_frac = 'Data/Hip/HipF&NF/valid/fractured'
hipValid_nfrac = 'Data/Hip/HipF&NF/valid/nonfractured'

"""
Find the count of all files
"""

# Train
HipTrain_frac_files = [os.path.join(root, f) for root, _, files in os.walk(hipTrain_frac) for f in files]

HipTrain_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(hipTrain_nfrac) for f in files]

# Test
HipTest_frac_files = [os.path.join(root, f) for root, _, files in os.walk(hipTest_frac) for f in files]

HipTest_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(hipTest_nfrac) for f in files]

# Valid
HipValid_frac_files = [os.path.join(root, f) for root, _, files in os.walk(hipValid_frac) for f in files]

HipValid_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(hipValid_nfrac) for f in files]

print(f"""
        Hip Files
        
        Train Frac Files: {len(HipTrain_frac_files)}
        Train NonFrac Files: {len(HipTrain_nfrac_files)}
        
        Test Frac Files: {len(HipTest_frac_files)}
        Test NonFrac Files: {len(HipTest_nfrac_files)}
        
        Valid Frac Files: {len(HipValid_frac_files)}
        Valid Non Files: {len(HipValid_nfrac_files)}
        """)

""" 
Result Hip Files Before Frac Atlas Edit
        
Train Frac Files: 1893
Train NonFrac Files: 81
        
Test Frac Files: 75
Test NonFrac Files: 12
        
Valid Frac Files: 140
Valid Non Files: 23
"""


