import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional

class KGDataLoader:
    """
    Handles loading, cleaning, and caching of the Knowledge Graph dataset.
    """
    def __init__(self, base_dir: str = '/app/output', slide_name: str = 'TENX96'):
        self.base_dir = base_dir
        self.slide_name = slide_name
        self.kg_dir = os.path.join(base_dir, slide_name, 'knowledge_graph')
        
    def load_data(self, cache_path: Optional[str] = None, force_reload: bool = False) -> pd.DataFrame:
        """
        Loads data either from a pickle cache or by crawling the KG directory.
        """
        if cache_path and os.path.exists(cache_path) and not force_reload:
            print(f"Loading cached dataset from {cache_path}...")
            df = pd.read_pickle(cache_path)
            print(f"Loaded {len(df)} patches from cache.")
            return self._validate_cohort(df)
            
        print("Crawling Knowledge Graph directory (this may take 1-2 minutes)...")
        df = self._crawl_kg_recursive()
        
        if cache_path:
            print(f"Saving cache to {cache_path}...")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_pickle(cache_path)
            
        return self._validate_cohort(df)

    def _crawl_kg_recursive(self) -> pd.DataFrame:
        patches = []
        clusters_dir = os.path.join(self.kg_dir, 'clusters')
        
        if not os.path.exists(clusters_dir):
            raise FileNotFoundError(f"Clusters directory not found at {clusters_dir}")
            
        for cluster_name in tqdm(sorted(os.listdir(clusters_dir)), desc="Loading Clusters"):
            cluster_path = os.path.join(clusters_dir, cluster_name)
            if not os.path.isdir(cluster_path): continue
            
            rois_dir = os.path.join(cluster_path, 'rois')
            if not os.path.exists(rois_dir): continue
                
            for roi_name in os.listdir(rois_dir):
                roi_path = os.path.join(rois_dir, roi_name)
                patches_dir = os.path.join(roi_path, 'patches')
                if not os.path.exists(patches_dir): continue
                    
                for patch_name in os.listdir(patches_dir):
                    patch_full_path = os.path.join(patches_dir, patch_name)
                    info_file = os.path.join(patch_full_path, 'info.json')
                    
                    if os.path.exists(info_file):
                        try:
                            with open(info_file, 'r') as f: info = json.load(f)
                            
                            # Load Embeddings
                            embeddings = self._load_embeddings(patch_full_path)
                            
                            # Extract CUI Data
                            cui_data = self._extract_cui_data(info)
                            
                            record = {
                                'patch_id': patch_name,
                                'roi_id': roi_name,
                                'cluster_id': cluster_name,
                                'x': info.get('x'),
                                'y': info.get('y'),
                                'spatial_label': info.get('cluster_id'),
                                **embeddings,
                                **cui_data
                            }
                            
                            # Flatten Gene Expression
                            gexpr = info.get('gene_expression', {})
                            # Load both sets
                            genes_top = gexpr.get('top_genes', {})
                            genes_marked = gexpr.get('marked_genes', {})
                            
                            # Merge (Marked overwrites Top if duplicate, though likely same value)
                            all_genes = {**genes_top, **genes_marked}
                            
                            for k, v in all_genes.items():
                                record[f'tgt_{k}'] = v
                                
                            patches.append(record)
                        except Exception as e:
                            # print(f"Error reading {patch_name}: {e}") # Silent for speed
                            pass
                            
        return pd.DataFrame(patches)

    def _load_embeddings(self, path: str) -> dict:
        txt_p = os.path.join(path, 'text_embedding.npy')
        img_p = os.path.join(path, 'image_embedding.npy')
        
        return {
            'text_embedding': np.load(txt_p) if os.path.exists(txt_p) else None,
            'image_embedding': np.load(img_p) if os.path.exists(img_p) else None
        }

    def _extract_cui_data(self, info: dict) -> dict:
        roi_ctx = info.get('roi_context', {})
        # Flatten lists safely
        raw_ctx = (roi_ctx.get('roi_concepts', []) or []) + (roi_ctx.get('patch_concepts', []) or [])
        
        all_cuis = list(set([c.get('cui') for c in raw_ctx if isinstance(c, dict) and 'cui' in c]))
        cui_map = {c.get('cui'): c.get('name') for c in raw_ctx if isinstance(c, dict) and 'cui' in c}
        
        return {
            'cuis': all_cuis,
            'cui_map': cui_map
        }
        
    def _validate_cohort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters for valid multi-modal instances."""
        initial_len = len(df)
        
        # Debug Stats
        n_img = df['image_embedding'].notna().sum() if 'image_embedding' in df.columns else 0
        print(f"DEBUG: Found {n_img}/{initial_len} patches with Image Embeddings.")
        
        if 'cuis' in df.columns:
            n_cui = df['cuis'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
            print(f"DEBUG: Found {n_cui}/{initial_len} patches with Valid CUIs.")
            
            # Print sample if no CUIs found
            if n_cui == 0 and initial_len > 0:
                print("DEBUG: Sample 'cuis' entry (Head):", df['cuis'].head().tolist())
                print("DEBUG: Sample 'roi_context' raw info keys (check loading logic).")

        df = df.dropna(subset=['image_embedding']).copy()
        
        # Ensure 'cuis' is a list and not empty
        def is_valid_cui(x):
            return isinstance(x, list) and len(x) > 0
            
        df = df[df['cuis'].apply(is_valid_cui)].reset_index(drop=True)
        
        print(f"Validation: {len(df)} valid patches (from {initial_len} total).")
        return df
