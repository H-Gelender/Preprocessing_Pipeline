
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class HierarchicalBioSwarm(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_genes=1000, num_heads=4):
        super().__init__()
        
        # --- Level 1: Patch -> ROI ---
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.roi_gat = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        
        # --- Level 2: ROI -> Cluster ---
        # Input to this is the aggregated ROI embedding + ROI semantic features (optional)
        # For now, we reuse the hidden dimension
        self.cluster_gat = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        
        # --- Predictor ---
        # Predicts gene expression from the ROI embedding (Local Resolution)
        self.gene_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_genes)
        )
        
    def forward(self, x_patch, edge_index_patch, batch_patch_to_roi):
        """
        x_patch: (N_patches, D_in)
        edge_index_patch: (2, E_patch) - Adjacency between patches (spatial)
        batch_patch_to_roi: (N_patches,) - Mapping each patch to its parent ROI index
        """
        
        # 1. Patch Level Processing
        h_patch = self.patch_encoder(x_patch)
        
        # Graph Convolution on Patches (if we had patch-level graph, 
        # but here we might just aggregate to ROI directly if edges are missing)
        # Let's assume we have a patch graph.
        h_patch = self.roi_gat(h_patch, edge_index_patch)
        
        # 2. Pooling to ROI Level (Hierarchy)
        # Global mean pool, but segmented by ROI ID
        # batch_patch_to_roi acts as the 'batch' index for pooling
        h_roi = global_mean_pool(h_patch, batch_patch_to_roi)
        
        # 3. Prediction (Per ROI)
        # We predict gene expression FOR THE ROI based on its constituent patches
        gene_pred = self.gene_predictor(h_roi)
        
        return gene_pred

