import h5py
import os

h5_path = '../hest_data/patches/TENX96.h5'

if os.path.exists(h5_path):
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"Keys: {list(f.keys())}")
            for key in f.keys():
                print(f"--- {key} ---")
                print(f"Shape: {f[key].shape}")
                print(f"Type: {f[key].dtype}")
                if len(f[key]) > 0:
                    print(f"First element sample: {f[key][0]}")
    except Exception as e:
        print(f"Error reading H5: {e}")
else:
    print(f"File not found: {h5_path}")
