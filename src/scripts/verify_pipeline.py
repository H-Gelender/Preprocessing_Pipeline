
import numpy as np
import h5py
from src.rl.env_bio_swarm import BioSwarmEnv

def check_structure():
    print("--- HDF5 Structure Check ---")
    with h5py.File('/app/output/TENX96_hierarchy.h5', 'r') as f:
        print("Clusters:", len(f['clusters'].keys()))
        print("ROIs:", len(f['rois'].keys()))
        print("Patches:", len(f['patches'].keys()))
        
        # Check linkage
        c0 = f['clusters']['cluster_0']
        print("Cluster 0 Child ROIs:", c0.attrs.get('child_rois', 'MISSING'))

def check_env():
    print("\n--- RL Environment Check ---")
    env = BioSwarmEnv()
    obs, _ = env.reset()
    print("Initial Obs Shape:", obs.shape)
    print("Current Node:", env.current_node_path)
    
    # Try Zoom
    print("Action: ZOOM_IN (Index 6)")
    obs, reward, term, trunc, _ = env.step(6) 
    print("New Node:", env.current_node_path)
    print("New Level:", env.current_level)
    print("Reward:", reward)
    
    if env.current_level == 1:
        print("Zoom Successful!")
    else:
        print("Zoom Failed.")
        
    env.close()

if __name__ == "__main__":
    try:
        check_structure()
        check_env()
    except Exception as e:
        print(f"Verification Failed: {e}")
