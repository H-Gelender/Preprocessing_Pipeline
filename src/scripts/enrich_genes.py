
import os
import json
import numpy as np
import scanpy as sc
from scipy.spatial import cKDTree
from tqdm import tqdm

def enrich_genes():
    slide_name = 'TENX96'
    h5ad_path = f'/app/hest_data/st/{slide_name}.h5ad'
    kg_dir = f'/app/output/{slide_name}/knowledge_graph'
    
    if not os.path.exists(h5ad_path):
        print(f"ST file not found: {h5ad_path}")
        return

    print(f"Loading ST data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    st_coords = adata.obsm['spatial']
    
    # Check bounds to ensure coordinate systems match (roughly)
    print(f"ST bounds: X[{st_coords[:,0].min()}-{st_coords[:,0].max()}] Y[{st_coords[:,1].min()}-{st_coords[:,1].max()}]")
    
    tree = cKDTree(st_coords)
    
    count = 0
    print(f"Scanning {kg_dir}...")
    
    patches_to_update = []
    
    for root, dirs, files in os.walk(kg_dir):
        if 'info.json' in files and 'patch_' in os.path.basename(root):
            patches_to_update.append(os.path.join(root, 'info.json'))
            
    print(f"Found {len(patches_to_update)} patches. Processing...")
    
    updated_count = 0
    
    for path in tqdm(patches_to_update):
        with open(path, 'r') as f: info = json.load(f)
        
        # If already has gene expression, skip? Maybe overwrite to be sure.
        # if 'gene_expression' in info: continue
        
        # Query nearest spot
        # Note: coords in info.json might be (x,y) or (y,x). 
        # Builder used [info['x'], info['y']]. Assuming correct.
        if 'x' not in info: continue
        
        dist, idx = tree.query([info['x'], info['y']])
        
        # Distance threshold (pixels). 
        # If 10X Visium, spacing is ~100px. 500px is safe.
        # Distance threshold
        if dist < 500: 
            # 1. Top 50 Genes (Expression)
            row = adata.X[idx].toarray().flatten() if hasattr(adata.X[idx], 'toarray') else adata.X[idx].flatten()
            
            top_idx = np.argsort(row)[-50:]
            top_genes = {adata.var_names[i]: float(row[i]) for i in top_idx if row[i] > 0}
            
            # 2. Marked Genes (Targeted Panel)
            marked_list = ["ESR1", "KRT8", "EGFR", "ERBB2", "MKI67", "KRT7", "PGR", "KRT5", "KRT14"]
            marked_genes = {}
            for gene in marked_list:
                if gene in adata.var_names:
                    gene_idx = adata.var_names.get_loc(gene)
                    marked_genes[gene] = float(row[gene_idx])
            
            expr = {
                'mapped_spot_idx': int(idx), 
                'distance': float(dist),
                'top_genes': top_genes,
                'marked_genes': marked_genes
            }
            
            info['gene_expression'] = expr
            
            with open(path, 'w') as f: json.dump(info, f, indent=2)
            updated_count += 1
            
    print(f"Enriched {updated_count}/{len(patches_to_update)} patches with gene expression.")

if __name__ == "__main__":
    enrich_genes()
