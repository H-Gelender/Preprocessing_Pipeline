
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import h5py
import torch

class BioSwarmEnv(gym.Env):
    """
    Hierarchical Environment for WSI Navigation.
    Levels: 0=Cluster, 1=ROI, 2=Patch
    """
    def __init__(self, h5_path='/app/output/TENX96_hierarchy.h5'):
        super().__init__()
        self.h5_path = h5_path
        self.h5f = None
        
        # Observation Space: [Node_Feat (2048), Context (8), Level (1)]
        # We simplify to just Node Features for now
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2048,), dtype=np.float32)
        
        # Action Space: Discrete
        # 0..K: Move to neighbor K
        # K+1: Zoom In
        # K+2: Predict (Stop)
        self.max_neighbors = 6
        self.action_space = spaces.Discrete(self.max_neighbors + 2)
        
        self.current_level = 0
        self.current_node_path = None # String path in HDF5, e.g., 'clusters/cluster_0'
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, 'r')
            
        # Start at a random Cluster
        clusters = list(self.h5f['clusters'].keys())
        start_cluster = np.random.choice(clusters)
        self.current_node_path = f"clusters/{start_cluster}"
        self.current_level = 0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        # Fetch current node embedding
        # In HDF5 builder, clusters are datasets, ROIs are groups with dataset 'image_embedding'
        # This divergence needs handling.
        
        if self.current_level == 0:
            # Cluster: We need a representation. 
            # The builder just stored ID. We need the mean.
            # Lazy fix: Return zeros for now or fetch ROI mean
            return np.zeros(2048, dtype=np.float32)
            
        elif self.current_level == 1:
            # ROI
            node = self.h5f[self.current_node_path]
            return node['image_embedding'][:]
            
        elif self.current_level == 2:
            # Patch
            node = self.h5f[self.current_node_path]
            return node[:] # Dataset itself
            
        return np.zeros(2048, dtype=np.float32)

    def step(self, action):
        terminated = False
        truncated = False
        reward = -0.01 # Time penalty
        
        # Actions
        zoom_action = self.max_neighbors
        predict_action = self.max_neighbors + 1
        
        if action == zoom_action:
            if self.current_level < 2:
                # Zoom In Logic
                # 1. Find children of current node
                children = self._get_children(self.current_node_path)
                if children:
                    # Deterministic Zoom to center child or random?
                    # For RL, 'Zoom' usually means entering the layer, then 'Move' selects specific node?
                    # Or 'Zoom' transitions to the graph of children.
                    # Let's say we land on the first child.
                    self.current_node_path = children[0]
                    self.current_level += 1
                else:
                    reward = -0.1 # Invalid zoom
            else:
                reward = -0.1 # Cannot zoom further
                
        elif action == predict_action:
            if self.current_level == 2:
                # Prediction logic
                # Compare with Ground Truth (which we need to fetch)
                reward = 1.0 # Placeholder
                terminated = True
            else:
                reward = -0.5 # Premature prediction
                
        else: # Move
            # Neighbor logic (Requires Adjacency Matrix)
            pass
            
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_children(self, node_path):
        # Fetch children paths based on attributes in HDF5
        node = self.h5f[node_path]
        if self.current_level == 0:
            # Cluster -> ROIs
            # Using 'child_rois' attribute added in updated builder script
            return [f"rois/{r}" for r in node.attrs.get('child_rois', [])]
        if self.current_level == 1:
             # ROI -> Patches
             return [f"patches/{p}" for p in node.attrs.get('child_patches', [])]
        return []

    def close(self):
        if self.h5f:
            self.h5f.close()
