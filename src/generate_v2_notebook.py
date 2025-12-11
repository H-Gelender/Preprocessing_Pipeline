
import nbformat as nbf
import os

def generate_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = []

    # Title
    nb.cells.append(nbf.v4.new_markdown_cell("""
# Interactive Multi-Modal Knowledge Graph Analysis
**dataset:** TENX96
**Modalities:** Image (Texture) + CUI (Semantics) + Gene (Transcriptomics)

This notebook provides a complete interactive analysis suite for:
1.  **Fusion Clustering**: Combining Morphology and Semantics.
2.  **Gene Integration**: Analyzing spatial transcriptomics and specific target markers.
3.  **Cross-Modality Comparison**: Confusion matrices and similarity heatmaps between Image, CUI, and Gene clusters.
4.  **Spatial Mapping**: Discrete clusters and continuous expression maps.
    """))

    # Block 1: Imports
    nb.cells.append(nbf.v4.new_code_cell("""
%load_ext autoreload
%autoreload 2

import os
import sys
# Add /app/src to path so we can import 'scripts' package
sys.path.append('/app/src')

import pandas as pd
import numpy as np
from scripts import KGDataLoader, MultiModalFusion, KGVisualizer

# Initialize Modules
base_dir = '/app/output'
loader = KGDataLoader(base_dir=base_dir, slide_name='TENX96')
fusion = MultiModalFusion(n_clusters=8)
viz = KGVisualizer()
    """))

    # Block 2: Load Data
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Load Data & Pre-Processing"))
    nb.cells.append(nbf.v4.new_code_cell("""
# Load Dataset (Force reload to pick up new Gene Enrichment data)
CACHE_PATH = '/app/scratchpad/dataset_cache.pkl'
df = loader.load_data(cache_path=CACHE_PATH, force_reload=True)

# Basic Stats
print(f"Loaded {len(df)} patches.")
    """))

    # Block 3: Modality Execution
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Execute Modality Clustering"))
    nb.cells.append(nbf.v4.new_code_cell("""
# A. Joint Fusion (Image + CUI)
df, X_fusion = fusion.process_fusion(df)

# B. Unimodal Clustering (Image Only, CUI Only) - Required for Comparison
df = fusion.compute_unimodal_clusters(df, n_clusters=8)

# C. Gene Expression Analysis (Top Expressed + Marked Panel)
marked_list = ["ESR1", "KRT8", "EGFR", "ERBB2", "MKI67", "KRT7", "PGR", "KRT5", "KRT14", "CD3D", "CD8A", "CD19"]
df, X_gene, gene_prof, marked_prof = fusion.process_gene_analysis(df, n_clusters=8, marked_genes=marked_list)
    """))

    # Block 4: Spatial Maps
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Spatial Cluster Maps (Discrete)"))
    nb.cells.append(nbf.v4.new_code_cell("""
# 1. Joint (Image + Concept)
viz.plot_spatial(df, 'joint_cluster', 'Modality: Joint (Image + CUI)').show()

# 2. Image Only (Texture)
viz.plot_spatial(df, 'img_cluster', 'Modality: Image Only').show()

# 3. Gene Only (Expression)
viz.plot_spatial(df, 'gene_cluster', 'Modality: Gene Expression').show()

# 4. CUI Only (Semantics)
viz.plot_spatial(df, 'cui_cluster', 'Modality: CUI Concepts').show()
    """))

    # Block 5: Confusion Matrices
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Cross-Modality Overlap (Confusion Matrices)"))
    nb.cells.append(nbf.v4.new_code_cell("""
# A. Gene vs Image
mat_img = fusion.compute_cluster_overlap(df, 'gene_cluster', 'img_cluster')
viz.plot_cluster_overlap(mat_img, "Overlap: Gene vs Image (Raw Count)", normalize=False).show()
viz.plot_cluster_overlap(mat_img, "Overlap: Gene vs Image (Probability)", normalize=True).show()

# B. Gene vs CUI
mat_cui = fusion.compute_cluster_overlap(df, 'gene_cluster', 'cui_cluster')
viz.plot_cluster_overlap(mat_cui, "Overlap: Gene vs CUI (Raw Count)", normalize=False).show()

# C. Gene vs Joint
mat_joint = fusion.compute_cluster_overlap(df, 'gene_cluster', 'joint_cluster')
viz.plot_cluster_overlap(mat_joint, "Overlap: Gene vs Joint (Raw Count)", normalize=False).show()
    """))

    # Block 6: Profiles
    nb.cells.append(nbf.v4.new_markdown_cell("## 5. Biological Profiles"))
    nb.cells.append(nbf.v4.new_code_cell("""
# 1. Top Expressed Genes per Cluster
viz.plot_gene_profiles(gene_prof).show()

# 2. Marked Genes (Target Panel) per Cluster
viz.plot_gene_profiles(marked_prof).show()

# 3. Top Semantic Concepts (CUI) per Cluster
cui_prof = fusion.get_top_concepts_df(df, top_n=10)
viz.plot_concept_profiles(cui_prof).show()
    """))
    
    # Block 7: Similarity Heatmaps
    nb.cells.append(nbf.v4.new_markdown_cell("## 6. Global Similarity Comparison"))
    nb.cells.append(nbf.v4.new_code_cell("""
# Compute similarity matrices for all modalities
sims = fusion.compute_all_similarities(df)
viz.plot_similarity_comparison(sims).show()
    """))
    
    # Block 8: Continuous Spatial Maps
    nb.cells.append(nbf.v4.new_markdown_cell("## 7. Continuous Spatial Gene Expression"))
    nb.cells.append(nbf.v4.new_code_cell("""
# Loop through Marked Genes and plot intensity maps
print("--- Targeted Gene Maps ---")
for gene in marked_list:
    col = f'tgt_{gene}'
    if col in df.columns:
        viz.plot_continuous_spatial(df, col, title=f"Expression: {gene}").show()

# Find Top 3 Global Genes
print("--- Top Global Expressed Genes ---")
gene_cols = [c for c in df.columns if c.startswith('tgt_')]
if gene_cols:
    global_means = df[gene_cols].mean().sort_values(ascending=False)
    for col in global_means.head(3).index:
        name = col.replace('tgt_', '')
        viz.plot_continuous_spatial(df, col, title=f"Top Global: {name}").show()

    # Block 9: Marker Discrimination
    nb.cells.append(nbf.v4.new_markdown_cell("## 8. Marker Discrimination Analysis (Which Modality is Best?)"))
    nb.cells.append(nbf.v4.new_code_cell("""
# For each marked gene, compare how well Image vs CUI vs Joint vs Gene clusters separate it.
# Metric: ANOVA F-Value (Higher = Better Separation/Distinctness)

cluster_cols = ['img_cluster', 'cui_cluster', 'joint_cluster', 'gene_cluster']

print("--- Marker Discrimination Scores (F-Value) ---")
for gene in marked_list:
    col = f'tgt_{gene}'
    if col not in df.columns: continue
    
    # Compute Scores
    scores = {}
    for c_col in cluster_cols:
        scores[c_col] = fusion.compute_marker_discrimination(df, col, c_col)
    
    # Plot Comparison
    viz.plot_marker_comparison(df, col, cluster_cols, scores).show()
    """))

    # Block 10: Save (Renumbering)
    nb.cells.append(nbf.v4.new_markdown_cell("## 9. Export Results"))
    nb.cells.append(nbf.v4.new_code_cell("""
df.to_pickle('/app/scratchpad/final_dataset_augmented.pkl')
print("Saved augmented dataset to /app/scratchpad/final_dataset_augmented.pkl")
    """))

    # Write file
    out_path = '/app/src/scripts/interactive_analysis.ipynb'
    with open(out_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Notebook generated at {out_path}")

if __name__ == "__main__":
    generate_notebook()
