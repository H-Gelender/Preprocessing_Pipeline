
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import h5py
import pandas as pd
import numpy as np
import glob
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

H5_PATH = '/app/output/TENX96_hierarchy.h5'
LOG_DIR = '/app/output/rl_logs'
OUTPUT_HTML = '/app/output/bio_swarm_report.html'

def get_training_metrics():
    """Parse Tensorboard logs for metrics"""
    runs = glob.glob(os.path.join(LOG_DIR, '*'))
    if not runs: return None
    latest_run = max(runs, key=os.path.getmtime)
    event_acc = EventAccumulator(latest_run)
    event_acc.Reload()
    
    data = {'step': [], 'value_loss': [], 'reward': []}
    
    # Value Loss
    if 'train/value_loss' in event_acc.Tags()['scalars']:
        events = event_acc.Scalars('train/value_loss')
        data['step'] = [e.step for e in events] # Assuming aligned
        data['value_loss'] = [e.value for e in events]
        
    # Reward
    if 'rollout/ep_rew_mean' in event_acc.Tags()['scalars']:
        events = event_acc.Scalars('rollout/ep_rew_mean')
        # Steps might differ, so we might need separate DFs or interpolation.
        # For simplicity, let's just create separate traces.
        rew_steps = [e.step for e in events]
        rew_vals = [e.value for e in events]
        return data, (rew_steps, rew_vals)
        
    return data, ([], [])

def build_graph_figure():
    """Build 2D Network Graph using NetworkX + Plotly"""
    G = nx.Graph()
    
    with h5py.File(H5_PATH, 'r') as f:
        # Clusters
        for c_key in f['clusters'].keys():
            G.add_node(c_key, type='cluster', size=20, color='red')
            
            # Child ROIs
            child_rois = f['clusters'][c_key].attrs.get('child_rois', [])
            for r_ref in child_rois:
                rid = f"roi_{r_ref.split('_')[1]}"
                if rid in f['rois']:
                    G.add_node(rid, type='roi', size=10, color='green')
                    G.add_edge(c_key, rid)
                    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        if G.nodes[node]['type'] == 'cluster':
            node_color.append('red')
        else:
            node_color.append('green')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=15,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text='Bio-Swarm Hierarchy (Static Snapshot)', font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

def generate():
    # 1. Graph
    fig_graph = build_graph_figure()
    
    # 2. Metrics
    metrics, (rew_steps, rew_vals) = get_training_metrics()
    
    fig_loss = go.Figure()
    if metrics['step']:
        fig_loss.add_trace(go.Scatter(x=metrics['step'], y=metrics['value_loss'], mode='lines', name='Value Loss'))
    fig_loss.update_layout(title='Training Value Loss', xaxis_title='Steps', yaxis_title='Loss')
    
    fig_reward = go.Figure()
    if rew_steps:
        fig_reward.add_trace(go.Scatter(x=rew_steps, y=rew_vals, mode='lines', name='Mean Reward', line=dict(color='orange')))
    fig_reward.update_layout(title='Training Reward', xaxis_title='Steps', yaxis_title='Reward')
    
    # Combine to HTML
    with open(OUTPUT_HTML, 'w') as f:
        f.write("<html><head><title>Bio-Swarm Report</title></head><body>")
        f.write("<h1 style='text-align:center; font-family:sans-serif'>Bio-Swarm Verification Report</h1>")
        
        f.write(fig_graph.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig_loss.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig_reward.to_html(full_html=False, include_plotlyjs='cdn'))
        
        f.write("</body></html>")
        
    print(f"Report saved to {OUTPUT_HTML}")

if __name__ == "__main__":
    generate()
