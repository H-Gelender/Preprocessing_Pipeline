"""
Knowledge Graph Multi-Modal Analysis Pipeline
Located in: src/analysis/kg_analysis_pipeline.py

This pipeline performs comprehensive analysis of the HEST Knowledge Graph by integrating
four distinct modalities:
1. Image Embeddings (ResNet/HistoEncoder - 2048 dim)
2. Text Embeddings (VLM Analysis via arctic-l-medembed - 1024 dim)
3. Targeted Gene Expression (Breast Cancer Panel)
4. Top Expressed Genes (Data-driven high variance genes)

Outputs:
- UMAP projections for all modalities
- Cluster consistency metrics (ARI, NMI)
- Cross-modal correlation analysis (Image ↔ Text, Text ↔ Genes)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
import umap
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class KGAnalysisPipeline:
    def __init__(self, slide_name, output_base_dir='../../output'):
        self.slide = slide_name
        self.base_dir = output_base_dir
        self.kg_dir = os.path.join(output_base_dir, slide_name, 'knowledge_graph')
        self.features_dir = os.path.join(output_base_dir, slide_name, 'features')
        self.results_dir = os.path.join(output_base_dir, slide_name, 'analysis_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Data containers
        self.df = None
        self.features = {}
        self.n_clusters = 8
        self.target_genes = ['ESR1', 'PGR', 'ERBB2', 'MKI67', 'EGFR', 'KRT5', 'KRT7', 'KRT8', 'KRT14']

    def load_data(self):
        """Load patches and embeddings from disk"""
        print(f"Loading Knowledge Graph data for {self.slide}...")
        patches = []
        clusters_dir = os.path.join(self.kg_dir, 'clusters')
        
        # Walk through KG structure
        for cluster_name in tqdm(sorted(os.listdir(clusters_dir)), desc="Parsing Clusters"):
            rois_dir = os.path.join(clusters_dir, cluster_name, 'rois')
            if not os.path.exists(rois_dir): continue
            
            for roi_name in os.listdir(rois_dir):
                patches_dir = os.path.join(rois_dir, roi_name, 'patches')
                if not os.path.exists(patches_dir): continue
                
                for patch_name in os.listdir(patches_dir):
                    patch_path = os.path.join(patches_dir, patch_name)
                    info_path = os.path.join(patch_path, 'info.json')
                    text_emb_path = os.path.join(patch_path, 'text_embedding.npy')
                    
                    if os.path.exists(info_path):
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        
                        gene_expr = info.get('gene_expression', {}) or {}
                        text_emb = np.load(text_emb_path) if os.path.exists(text_emb_path) else None
                        
                        patches.append({
                            'patch_id': info['patch_id'],
                            'cluster_id': info['cluster_id'],
                            'x': info['x'], 'y': info['y'],
                            'panel_genes': gene_expr.get('panel_genes', {}),
                            'top_genes': gene_expr.get('top_expressed', []),
                            'text_embedding': text_emb
                        })
        
        self.df = pd.DataFrame(patches)
        print(f"Loaded {len(self.df)} patches")
        self._match_image_embeddings()
        self._create_feature_matrices()

    def _match_image_embeddings(self):
        """Match patches to pre-computed image feature embeddings via coordinates"""
        print("Matching image embeddings...")
        img_feats = np.load(os.path.join(self.features_dir, 'features.npy'))
        img_coords = np.load(os.path.join(self.features_dir, 'coords.npy'))
        
        tree = cKDTree(img_coords)
        dists, idxs = tree.query(self.df[['x', 'y']].values)
        
        # Filter valid matches (within 50px tolerance) and valid text embeddings
        valid = (dists < 50) & self.df['text_embedding'].notna()
        self.df = self.df[valid].copy()
        self.df['img_idx'] = idxs[valid]
        
        # Store full image embeddings matrix for matched patches
        self.features['Image'] = img_feats[self.df['img_idx'].values]
        print(f"Patches with valid multi-modal data: {len(self.df)}")

    def _create_feature_matrices(self):
        """Construct feature matrices for all modalities"""
        print("Creating feature matrices...")
        
        # 1. Text Embeddings
        self.features['Text'] = np.vstack(self.df['text_embedding'].values)
        
        # 2. Targeted Genes
        self.features['Target'] = np.array([
            [row['panel_genes'].get(g, 0) for g in self.target_genes] 
            for _, row in self.df.iterrows()
        ])
        
        # 3. Top Expressed Genes
        all_top = set()
        for genes in self.df['top_genes']:
            if genes: all_top.update([g['gene'] for g in genes])
        
        top_list = sorted(all_top)
        top_map = {g: i for i, g in enumerate(top_list)}
        
        X_top = np.zeros((len(self.df), len(top_list)))
        for i, genes in enumerate(self.df['top_genes']):
            for g in genes:
                if g['gene'] in top_map:
                    X_top[i, top_map[g['gene']]] = g['expression']
        self.features['Top_Genes'] = X_top
        
        self.labels_true = self.df['cluster_id'].values

    def compute_umap(self):
        """Compute UMAP projections for all modalities"""
        print("Computing UMAP projections...")
        projections = {}
        for name, X in self.features.items():
            print(f"  Projecting {name} ({X.shape[1]} dims)...")
            X_scaled = StandardScaler().fit_transform(X)
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            projections[name] = reducer.fit_transform(X_scaled)
        
        return projections

    def analyze_consistency(self):
        """Analyze clustering consistency across modalities"""
        print("Analyzing cluster consistency...")
        
        # Re-cluster each modality
        labels = {'Original': self.labels_true}
        for name, X in self.features.items():
            X_scaled = StandardScaler().fit_transform(X)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
            labels[name] = kmeans.fit_predict(X_scaled)
        
        # Compute pairwise metrics
        modalities = list(labels.keys())
        n = len(modalities)
        ari = pd.DataFrame(index=modalities, columns=modalities, dtype=float)
        
        for n1 in modalities:
            for n2 in modalities:
                ari.loc[n1, n2] = adjusted_rand_score(labels[n1], labels[n2])
        
        return ari

    def analyze_correlations(self):
        """Compute cross-modal correlations"""
        print("Computing correlations...")
        correlations = {}
        
        # Image <-> Text (canonical correlation via PCA)
        pca_img = PCA(n_components=10).fit_transform(StandardScaler().fit_transform(self.features['Image']))
        pca_txt = PCA(n_components=10).fit_transform(StandardScaler().fit_transform(self.features['Text']))
        
        img_txt_corr = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                img_txt_corr[i, j] = spearmanr(pca_img[:, i], pca_txt[:, j])[0]
        correlations['Image_Text_PCA'] = img_txt_corr
        
        return correlations

    def run(self):
        """Execute full pipeline"""
        self.load_data()
        
        # 1. Projections
        projections = self.compute_umap()
        np.savez(os.path.join(self.results_dir, 'umap_projections.npz'), **projections)
        
        # 2. Consistency
        ari_matrix = self.analyze_consistency()
        ari_matrix.to_csv(os.path.join(self.results_dir, 'cluster_consistency_ari.csv'))
        print("\nCluster Consistency (ARI):")
        print(ari_matrix)
        
        # 3. Correlations
        corrs = self.analyze_correlations()
        np.savez(os.path.join(self.results_dir, 'correlations.npz'), **corrs)
        
        print(f"\n✅ Analysis complete. Results saved to {self.results_dir}")

if __name__ == "__main__":
    pipeline = KGAnalysisPipeline('TENX96')
    pipeline.run()
