
import h5py
import numpy as np
import json
import os
from tqdm import tqdm
from glob import glob

base_path = '/app/output/TENX96'
output_file = '/app/output/TENX96_hierarchy.h5'

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_npy(path):
    if os.path.exists(path):
        return np.load(path).flatten()
    return None

def build_hierarchy():
    print(f"Building hierarchy from {base_path}...")
    
    with h5py.File(output_file, 'w') as h5f:
        # Create groups
        grp_clusters = h5f.create_group('clusters')
        grp_rois = h5f.create_group('rois')
        grp_patches = h5f.create_group('patches')
        
        # --- 1. CLUSTERS ---
        cluster_dirs = sorted(glob(os.path.join(base_path, 'knowledge_graph', 'clusters', 'cluster_*')))
        print(f"Found {len(cluster_dirs)} clusters.")
        
        for c_path in tqdm(cluster_dirs, desc="Clusters"):
            c_info = load_json(os.path.join(c_path, 'info.json'))
            cid = c_info['cluster_id']
            c_name = f"cluster_{cid}"
            
            # Create Cluster Node
            dset = grp_clusters.create_dataset(c_name, data=np.array([cid]))
            # Attributes
            dset.attrs['num_rois'] = c_info.get('num_rois', 0)
            dset.attrs['num_patches'] = c_info.get('num_patches', 0)
            
            # --- 2. ROIS within Cluster ---
            roi_dirs = sorted(glob(os.path.join(c_path, 'rois', 'roi_*')))
            roi_refs = []
            
            for r_path in roi_dirs:
                r_info = load_json(os.path.join(r_path, 'info.json'))
                rid = r_info['roi_id']
                r_name = f"roi_{rid}"
                
                # Features
                img_emb = load_npy(os.path.join(r_path, 'image_embedding.npy'))
                cui_emb = load_npy(os.path.join(r_path, 'concept_embedding.npy')) # Using concept_embedding as CUI vector proxy
                
                # Create ROI Node
                # We store the aggregated gene expression as the target
                gene_expr = r_info.get('gene_expression', {}).get('aggregated', {})
                # Simplify: just store the mean of specific genes or raw dict as string? 
                # For HDF5, let's store key targets as attributes
                
                if img_emb is not None:
                     # Create dataset for ROI features (Fixed vector if possible, or just create group)
                     # Let's create a group for the ROI to store datasets
                     roi_grp = grp_rois.create_group(r_name)
                     roi_grp.attrs['parent_cluster'] = c_name
                     roi_grp.attrs['centroid'] = r_info.get('centroid', [0,0])
                     
                     roi_grp.create_dataset('image_embedding', data=img_emb)
                     if cui_emb is not None:
                        roi_grp.create_dataset('cui_embedding', data=cui_emb)
                     
                     # Store Gene Targets
                     genes_to_store = ['ESR1', 'ERBB2', 'KRT8', 'KRT7'] # Examples
                     for g in genes_to_store:
                         if g in gene_expr:
                             roi_grp.attrs[f'target_{g}'] = gene_expr[g]['mean']

                # --- 3. PATCHES within ROI ---
                patch_dirs = sorted(glob(os.path.join(r_path, 'patches', 'patch_*')))
                
                patch_refs = []
                
                for p_path in patch_dirs:
                    p_info = load_json(os.path.join(p_path, 'info.json'))
                    pid = p_info['patch_id']
                    p_name = f"patch_{pid}"
                    
                    # Features
                    p_img = load_npy(os.path.join(p_path, 'image_embedding.npy'))
                    
                    if p_img is not None:
                        # Create Patch Dataset
                        p_dset = grp_patches.create_dataset(p_name, data=p_img)
                        p_dset.attrs['parent_roi'] = r_name
                        p_dset.attrs['coords'] = p_info.get('coords', [0,0])
                        
                        patch_refs.append(p_name)
                
                # Link patches to ROI
                roi_grp.attrs['child_patches'] = patch_refs
                roi_refs.append(r_name)

            dset.attrs['child_rois'] = roi_refs

    print(f"Hierarchy saved to {output_file}")

if __name__ == "__main__":
    build_hierarchy()
