
import os
import json
import numpy as np
import shutil
from tqdm import tqdm

def distribute_features():
    features_path = '/app/output/TENX96/features/features.npy'
    if not os.path.exists(features_path):
        print("Features file not found!")
        return

    print("Loading features...")
    features = np.load(features_path)
    print(f"Loaded features shape: {features.shape}")
    
    base_dir = '/app/output/TENX96/knowledge_graph'
    
    count = 0
    # Walk the directory to find info.json
    for root, dirs, files in os.walk(base_dir):
        if 'info.json' in files:
            try:
                with open(os.path.join(root, 'info.json'), 'r') as f:
                    info = json.load(f)
                
                if 'patch_id' in info:
                    pidx = info['patch_id']
                    if pidx < len(features):
                        emb = features[pidx]
                        np.save(os.path.join(root, 'image_embedding.npy'), emb)
                        count += 1
            except Exception as e:
                print(f"Error processing {root}: {e}")
                
    print(f"Distributed {count} embeddings.")

if __name__ == "__main__":
    distribute_features()
