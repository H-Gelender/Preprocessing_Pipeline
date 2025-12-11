
import h5py
import numpy as np

path = '/app/output/TENX96_hierarchy.h5'

def repair():
    with h5py.File(path, 'r+') as f:
        valid_rois = set(f['rois'].keys())
        print(f"Valid ROIs: {len(valid_rois)}")
        
        clusters = f['clusters']
        for c_key in clusters.keys():
            dset = clusters[c_key]
            child_rois = dset.attrs.get('child_rois', [])
            
            # Filter
            clean_rois = [r for r in child_rois if r in valid_rois]
            
            if len(clean_rois) != len(child_rois):
                print(f"Fixing {c_key}: {len(child_rois)} -> {len(clean_rois)} ROIs")
                dset.attrs['child_rois'] = clean_rois

    print("Repair complete.")

if __name__ == "__main__":
    repair()
