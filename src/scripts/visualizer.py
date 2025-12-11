import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional

class KGVisualizer:
    """
    Generates interactive visualizations for the Knowledge Graph.
    """
    def __init__(self):
        px.defaults.template = "plotly_white"
        
    def plot_spatial(self, df: pd.DataFrame, color_col: str, title: str, hover_cols: list = None):
        """Generates a spatial scatterplot."""
        hover = hover_cols if hover_cols else ['patch_id']
        
        fig = px.scatter(
            df, x='x', y='y', color=color_col,
            title=title,
            hover_data=hover,
            width=900, height=800
        )
        fig.update_yaxes(autorange="reversed") # Image coord system
        fig.update_traces(marker=dict(size=4.5))
        return fig
        
    def plot_umap(self, df: pd.DataFrame, X_data: np.ndarray, color_col: str, title: str):
        """Computes and plots UMAP."""
        print(f"Computing UMAP for {title}...")
        
        # Scanpy UMAP
        adata = ad.AnnData(X_data)
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
        sc.tl.umap(adata, min_dist=0.1)
        coords = adata.obsm['X_umap']
        
        df['umap_1'] = coords[:, 0]
        df['umap_2'] = coords[:, 1]
        
        fig = px.scatter(
            df, x='umap_1', y='umap_2', color=color_col,
            title=f"UMAP: {title}",
            hover_data=['domain_desc'] if 'domain_desc' in df.columns else None,
            width=900, height=700
        )
        return fig
        return fig

    def compare_modalities(self, df: pd.DataFrame, X_img: np.ndarray, X_cui: np.ndarray, n_clusters=8):
        """Displays 2 Maps: one clustered by Image, one by CUI."""
        from sklearn.cluster import KMeans
        
        print("Clustering Image-only...")
        l_img = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_img)
        df['img_cluster'] = l_img.astype(str)
        
        print("Clustering CUI-only...")
        l_cui = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_cui)
        df['cui_cluster'] = l_cui.astype(str)
        
        fig1 = self.plot_spatial(df, 'img_cluster', 'Modality: Image Only (Texture)')
        fig2 = self.plot_spatial(df, 'cui_cluster', 'Modality: CUI Only (Concept)')
        
        return fig1, fig2

    def plot_concept_profiles(self, profile_df: pd.DataFrame):
        """Generates a faceted bar chart of Top Concepts per Cluster."""
        # Truncate long names for display
        profile_df = profile_df.copy()
        profile_df['Concept'] = profile_df['Concept'].apply(lambda x: (x[:35] + '...') if len(x) > 35 else x)
        
        fig = px.bar(
            profile_df, 
            x='Score', 
            y='Concept', 
            color='Cluster',
            facet_col='Cluster', 
            facet_col_wrap=2, # Fewer columns = Wider plots
            orientation='h',
            title='Top Concepts per Joint Cluster',
            height=1200, # Taller figure
            hover_data=['Score']
        )
        
        # Independent Y axes so labels are specific to each cluster
        fig.update_yaxes(matches=None, showticklabels=True, automargin=True)
        fig.update_xaxes(matches=None)
        
        # Invert y-axis and increase label space
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(l=150) # More space for labels
        )
        # Apply to all subplots
        fig.for_each_yaxis(lambda yaxis: yaxis.update(autorange="reversed"))
        
        return fig

    def plot_similarity_comparison(self, sims: dict):
        """Side-by-side heatmaps of available modalities with independent scaling."""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Determine active sims
        actives = [k for k in ['img', 'cui', 'gene'] if k in sims]
        titles = {
            'img': "Image Similarity (Texture)",
            'cui': "CUI Similarity (Concept)",
            'gene': "Gene Similarity (Expression)"
        }
        
        fig = make_subplots(
            rows=1, cols=len(actives), 
            subplot_titles=[titles[k] for k in actives],
            horizontal_spacing=0.15
        )
        
        for i, key in enumerate(actives):
            sim_df = sims[key]
            # Use independent color scaling (no zmin/zmax fixed)
            heatmap = go.Heatmap(
                z=sim_df.values,
                x=sim_df.columns, y=sim_df.index,
                colorscale='RdBu_r', 
                # zmin/zmax removed to allow auto-scaling per matrix
                colorbar=dict(x=1.005 if i == len(actives)-1 else None, title='Sim', thickness=15)
            )
            fig.add_trace(heatmap, row=1, col=i+1)
        
        fig.update_layout(height=500, width=500*len(actives), title_text="Inter-Cluster Similarity Comparison (Independent Scales)")
        return fig

    def plot_gene_profiles(self, profile_df: pd.DataFrame):
        """Generates a faceted bar chart of Top Genes per Cluster."""
        fig = px.bar(
            profile_df, 
            x='Expression', 
            y='Gene', 
            color='Cluster',
            facet_col='Cluster', 
            facet_col_wrap=2,
            orientation='h',
            title='Top Expressed Genes per Spatial Cluster',
            height=1200,
            hover_data=['Expression']
        )
        fig.update_yaxes(matches=None, showticklabels=True, automargin=True)
        fig.update_xaxes(matches=None)
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(l=100)
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(l=100)
        )
        fig.for_each_yaxis(lambda yaxis: yaxis.update(autorange="reversed"))
        return fig

    def plot_cluster_overlap(self, overlap_df: pd.DataFrame, title: str = "Cluster Overlap", normalize: bool = False):
        """Plots a heatmap of confusion matrix (Raw Counts or Normalized)."""
        import plotly.graph_objects as go
        
        if normalize:
            # Row-Normalize
            data_df = overlap_df.div(overlap_df.sum(axis=1), axis=0)
            fmt = ".2f"
            subtitle = "Row-Normalized Probability"
        else:
            # Raw Counts
            data_df = overlap_df
            fmt = "d"
            subtitle = "Raw Counts (Number of Patches)"
        
        fig = go.Figure(data=go.Heatmap(
            z=data_df.values,
            x=overlap_df.columns,
            y=overlap_df.index,
            text=overlap_df.values, 
            texttemplate="%{text}",
            colorscale='Mint',
            colorbar=dict(title='Count' if not normalize else 'Prob')
        ))
        
        fig.update_layout(
            title=f"{title} ({subtitle})",
            xaxis_title="Cluster Set B (Columns)",
            yaxis_title="Cluster Set A (Rows)",
            height=600, width=700,
            yaxis=dict(autorange="reversed")
        )
        return fig

    def plot_continuous_spatial(self, df: pd.DataFrame, column: str, title: str):
        """Plots a spatial map colored by a continuous variable (e.g. Gene Expression)."""
        # Ensure numeric
        if column not in df.columns:
            print(f"Column {column} not found.")
            return None
            
        fig = px.scatter(
            df, 
            x='x', 
            y='y', 
            color=column,
            title=title,
            color_continuous_scale='Viridis',
            hover_data=['patch_id'],
            width=900, height=700
        )
        
        # Match WSI coordinate system (Y starts from top)
        fig.update_yaxes(autorange='reversed')
        fig.update_layout(plot_bgcolor='white')
        return fig

    def plot_marker_comparison(self, df: pd.DataFrame, gene_col: str, cluster_cols: list, scores: dict):
        """Violin plots comparing gene distribution across different clustering modalities."""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        gene_name = gene_col.replace('tgt_', '')
        fig = make_subplots(
            rows=1, cols=len(cluster_cols),
            subplot_titles=[f"{c} (F={scores.get(c,0):.1f})" for c in cluster_cols]
        )
        
        for i, col in enumerate(cluster_cols):
            # Box/Violin for each cluster, for this gene
            fig.add_trace(
                go.Box(
                    y=df[gene_col],
                    x=df[col],
                    name=col,
                    boxpoints='outliers' # Only show outliers to save rendering time
                ),
                row=1, col=i+1
            )
            
        fig.update_layout(
            title=f"Distribution of {gene_name} by Modality (Higher F = Better Separation)",
            showlegend=False,
            height=500, width=500 * len(cluster_cols)
        )
        return fig
