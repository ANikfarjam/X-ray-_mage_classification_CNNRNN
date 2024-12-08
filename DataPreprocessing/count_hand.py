import os
import os.path as path

frac_dir = 'Data/Hand_Part_3/fractured'
nfrac_dir = 'Data/Hand_Part_3/nonfractured'

frac_files = [os.path.join(root, f) for root, _, files in os.walk(frac_dir) for f in files]
nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(nfrac_dir) for f in files]

print(len(frac_files))
print(len(nfrac_files))