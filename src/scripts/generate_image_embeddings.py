
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

def generate_embeddings(base_dir='/app/output'):
    print("Initializing ResNet50...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Pretrained ResNet
    model = models.resnet50(pretrained=True)
    # Remove classification layer (get 2048-d vector)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    
    # Transform
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print(f"Scanning {base_dir} for patches...")
    patches_to_process = []
    
    for root, dirs, files in os.walk(base_dir):
        if 'info.json' in files and 'image.png' in files:
            # Check if embedding needs generation
            emb_path = os.path.join(root, 'image_embedding.npy')
            if not os.path.exists(emb_path):
                patches_to_process.append(root)
    
    print(f"Found {len(patches_to_process)} patches missing image embeddings.")
    
    if not patches_to_process:
        print("All patches have embeddings. Exiting.")
        return

    # Process
    batch_size = 32
    for i in tqdm(range(0, len(patches_to_process), batch_size), desc="Generating Embeddings"):
        batch_paths = patches_to_process[i:i+batch_size]
        batch_imgs = []
        valid_indices = []
        
        for idx, p_dir in enumerate(batch_paths):
            try:
                img_path = os.path.join(p_dir, 'image.png')
                img = Image.open(img_path).convert('RGB')
                batch_imgs.append(preprocess(img))
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if not batch_imgs: continue
        
        # Batch Stack
        img_tensor = torch.stack(batch_imgs).to(device)
        
        with torch.no_grad():
            features = model(img_tensor)
            # Flatten: (B, 2048, 1, 1) -> (B, 2048)
            features = features.flatten(1).cpu().numpy()
            
        # Save
        for local_idx, feat_vec in zip(valid_indices, features):
            save_path = os.path.join(batch_paths[local_idx], 'image_embedding.npy')
            np.save(save_path, feat_vec)

    print("Done! Image embeddings generated.")

if __name__ == "__main__":
    generate_embeddings()
