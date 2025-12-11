
import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_cytoscape as cyto
import pandas as pd
import numpy as np
import h5py
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- 1. DATA LOADING ---
H5_PATH = '/app/output/TENX96_hierarchy.h5'
LOG_DIR = '/app/output/rl_logs'

def load_clusters():
    """Load initial clusters from HDF5"""
    elements = []
    with h5py.File(H5_PATH, 'r') as f:
        for c_key in f['clusters'].keys():
            # Node
            elements.append({
                'data': {'id': c_key, 'label': c_key, 'type': 'cluster', 'color': '#FF5733'},
                'position': {'x': np.random.randint(0, 500), 'y': np.random.randint(0, 500)}
            })
    return elements

def get_children_elements(node_id):
    """Fetch children (ROIs or Patches) for a node"""
    new_elements = []
    
    with h5py.File(H5_PATH, 'r') as f:
        if node_id.startswith('cluster'):
            # Fetch ROIs
            if node_id not in f['clusters']: return []
            child_rois = f['clusters'][node_id].attrs.get('child_rois', [])
            
            for roi in child_rois:
                rid = f"roi_{roi.split('_')[1]}" # Ensure format
                if rid in f['rois']: # Check validity
                     # Add Node
                     new_elements.append({
                        'data': {'id': rid, 'label': rid, 'type': 'roi', 'color': '#33FF57'}
                     })
                     # Add Edge
                     new_elements.append({
                         'data': {'source': node_id, 'target': rid}
                     })
                     
        elif node_id.startswith('roi'):
            # Fetch Patches (Limit to 10 for viz performance)
            if node_id not in f['rois']: return []
            child_patches = f['rois'][node_id].attrs.get('child_patches', [])[:10]
            
            for p in child_patches:
                 # Add Node
                 new_elements.append({
                    'data': {'id': p, 'label': '', 'type': 'patch', 'color': '#3357FF'}
                 })
                 # Add Edge
                 new_elements.append({
                     'data': {'source': node_id, 'target': p}
                 })
                 
    return new_elements

def parse_tensorboard():
    """Parse latest TB log to DF"""
    # Find latest run
    runs = glob.glob(os.path.join(LOG_DIR, '*'))
    if not runs: return pd.DataFrame()
    
    latest_run = max(runs, key=os.path.getmtime)
    event_acc = EventAccumulator(latest_run)
    event_acc.Reload()
    
    # Extract Scalar: rollout/ep_rew_mean or train/value_loss
    data = {}
    tags = event_acc.Tags()['scalars']
    
    for tag in ['train/value_loss', 'rollout/ep_rew_mean', 'train/loss']:
        if tag in tags:
            events = event_acc.Scalars(tag)
            data[tag] = [e.value for e in events]
            data['step'] = [e.step for e in events]
            
    return pd.DataFrame(data)

# --- 2. LAYOUT ---
app = dash.Dash(__name__)

default_stylesheet = [
    {'selector': 'node', 'style': {'label': 'data(label)', 'width': '30px', 'height': '30px'}},
    {'selector': '[type = "cluster"]', 'style': {'background-color': '#FF5733', 'shape': 'hexagon', 'width': 60, 'height': 60}},
    {'selector': '[type = "roi"]', 'style': {'background-color': '#33FF57', 'shape': 'triangle'}},
    {'selector': '[type = "patch"]', 'style': {'background-color': '#3357FF', 'width': 10, 'height': 10}},
    {'selector': 'edge', 'style': {'line-color': '#ccc'}}
]

app.layout = html.Div([
    html.H1("Bio-Swarm Neuro-Symbolic Dashboard", style={'textAlign': 'center'}),
    
    dcc.Tabs([
        # TAB 1: NAVIGATOR
        dcc.Tab(label='Knowledge Graph Navigator', children=[
            html.Div([
                cyto.Cytoscape(
                    id='cytoscape-graph',
                    elements=load_clusters(),
                    style={'width': '100%', 'height': '600px'},
                    layout={'name': 'cose'},
                    stylesheet=default_stylesheet
                )
            ], style={'border': '1px solid #ddd', 'padding': '10px'})
        ]),
        
        # TAB 2: TRAINING METRICS
        dcc.Tab(label='RL Training Metrics', children=[
            html.Div([
                html.Button('Refresh Logs', id='btn-refresh'),
                dcc.Graph(id='graph-loss'),
                dcc.Graph(id='graph-reward')
            ], style={'padding': '20px'})
        ])
    ])
])

# --- 3. CALLBACKS ---

@app.callback(
    Output('cytoscape-graph', 'elements'),
    Input('cytoscape-graph', 'tapNodeData'),
    State('cytoscape-graph', 'elements')
)
def expand_node(node_data, current_elements):
    if not node_data:
        return dash.no_update
        
    node_id = node_data['id']
    
    # Check if already expanded (naive check: look for edges starting from this node)
    # Better: check if we just added children.
    # For now, just append children. Cytoscape handles duplicates usually or we filter.
    
    children = get_children_elements(node_id)
    if not children:
        return dash.no_update
        
    # Dedup
    existing_ids = set(e['data']['id'] if 'source' not in e['data'] else (e['data']['source']+e['data']['target']) for e in current_elements)
    
    new_unique = []
    for el in children:
        uid = el['data']['id'] if 'source' not in el['data'] else (el['data']['source']+el['data']['target'])
        if uid not in existing_ids:
            new_unique.append(el)
            
    return current_elements + new_unique

@app.callback(
    [Output('graph-loss', 'figure'), Output('graph-reward', 'figure')],
    [Input('btn-refresh', 'n_clicks')]
)
def update_metrics(n):
    df = parse_tensorboard()
    
    fig_loss = {'data': [], 'layout': {'title': 'Value Loss'}}
    fig_reward = {'data': [], 'layout': {'title': 'Reward'}}
    
    if not df.empty:
        if 'train/value_loss' in df:
            fig_loss['data'] = [{'x': df['step'], 'y': df['train/value_loss'], 'type': 'line', 'name': 'Value Loss'}]
        if 'rollout/ep_rew_mean' in df:
             fig_reward['data'] = [{'x': df['step'], 'y': df['rollout/ep_rew_mean'], 'type': 'line', 'name': 'Mean Reward'}]
             
    return fig_loss, fig_reward

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
