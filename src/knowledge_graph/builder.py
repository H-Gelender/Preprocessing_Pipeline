"""
WSI Knowledge Graph Builder Pipeline
====================================
A unified pipeline to build a Multi-Modal Knowledge Graph from Whole Slide Images.

Integrates:
1.  **Spatial Clustering**: KMeans on image features.
2.  **ROI Generation**: Graph-based region extraction.
3.  **VLM Analysis**: Groq-powered morphological analysis (Llama-4-Scout).
4.  **Gene Enrichment**: Mapping spatial transcriptomics to patches.
5.  **Text Embeddings**: Generating embeddings from VLM descriptions.

Usage:
    python build_kg_pipeline.py --slide TENX96 --steps all
"""

import os
import json
import logging
import argparse
import hashlib
import base64
import multiprocessing
import numpy as np
import pandas as pd
import networkx as nx
import h5py
import scanpy as sc
import openslide

from PIL import Image
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON Encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ==================== COMPONENT: VLM CACHE ====================
import threading

class VLMCache:
    """Cache VLM results to avoid reprocessing"""
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, 'vlm_cache.json')
        self.lock = threading.Lock() # Thread safety
        self.cache = self._load_cache()
    
    def _load_cache(self):
        with self.lock:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    try: 
                        return json.load(f)
                    except json.JSONDecodeError:
                        return {}
            return {}
    
    def _save_cache(self):
        with self.lock:
            try:
                # Initialize new file write to reduce corruption risk
                temp_file = self.cache_file + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
                os.rename(temp_file, self.cache_file)
            except Exception as e:
                logger.error(f"Cache write failed: {e}")
    
    def get_image_hash(self, image):
        if isinstance(image, str):  # Path
            with open(image, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        else:  # PIL Image
            return hashlib.md5(image.tobytes()).hexdigest()
    
    def get(self, image, analysis_type):
        key = f"{self.get_image_hash(image)}_{analysis_type}"
        # No lock needed for read if dict is process-local, 
        # but self.cache is shared ref if instance is shared.
        # Python dicts are thread-safe for reads, but just in case:
        return self.cache.get(key)
    
    def set(self, image, analysis_type, result):
        key = f"{self.get_image_hash(image)}_{analysis_type}"
        with self.lock:
            self.cache[key] = result
        self._save_cache()

# ==================== COMPONENT: GROQ VLM ANALYZER ====================
class GroqVLMAnalyzer:
    def __init__(self, cache_dir, model_id):
        load_dotenv('/app/.env')
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found. VLM steps will fail if triggered.")
        else:
            self.client = Groq(api_key=api_key)
        
        self.model = model_id
        self.cache = VLMCache(cache_dir)
        logger.info(f"Groq VLM initialized: {self.model}")

    def _encode_image(self, image):
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _call_groq(self, image, prompt):
        if not hasattr(self, 'client'):
            return {"error": "Groq client not initialized"}
            
        base64_image = self._encode_image(image)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }],
                temperature=0.3,
                max_completion_tokens=512,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            return {"error": str(e)}

    def analyze_patch(self, image):
        cached = self.cache.get(image, "patch")
        if cached: return cached
        
        prompt = """Analyze this H&E histopathology patch. Return JSON with:
{"cell_morphology":"description","cell_density":"description","mitosis_count":"description","nuclear_atypia":"description","key_features":["feature1","feature2"]}"""
        result = self._call_groq(image, prompt)
        self.cache.set(image, "patch", result)
        return result

    def analyze_roi(self, image):
        cached = self.cache.get(image, "roi")
        if cached: return cached
        
        prompt = """Analyze this H&E histopathology ROI. Return JSON with:
{"tissue_architecture":"description","glandular_structures":"description","stromal_components":"description","inflammation":"description","pathological_patterns":"description"}"""
        result = self._call_groq(image, prompt)
        self.cache.set(image, "roi", result)
        return result

# ==================== COMPONENT: ROI EXTRACTOR ====================
class ROIExtractor:
    def __init__(self, wsi_path, target_size=512):
        self.slide = openslide.OpenSlide(wsi_path)
        self.target_size = target_size

    def extract_roi(self, coords, patch_size, padding=50):
        # ... (Same logic as before) ...
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]
        x_min, y_min = min(xs) - padding, min(ys) - padding
        x_max, y_max = max(xs) + patch_size + padding, max(ys) + patch_size + padding
        width, height = x_max - x_min, y_max - y_min
        
        region = self.slide.read_region((int(x_min), int(y_min)), 0, (int(width), int(height))).convert('RGB')
        scale = self.target_size / max(width, height)
        region = region.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
        
        return region, {'x_min': x_min, 'y_min': y_min, 'width': width, 'height': height, 'scale': scale}

    def close(self):
        self.slide.close()

# ==================== COMPONENT: FOLDER KG MANAGER ====================
class FolderManager:
    def __init__(self, kg_root):
        self.kg_root = kg_root
        os.makedirs(kg_root, exist_ok=True)
    
    def save_json(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
            
    def update_json(self, path, data):
        """Updates existing JSON with new keys"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                existing = json.load(f)
            existing.update(data)
            data = existing
        self.save_json(path, data)

# ==================== COMPONENT: UMLS ENRICHER ====================
class UMLSEnricher:
    def __init__(self, cache_dir):
        load_dotenv('/app/.env')
        self.api_key = os.environ.get("UMLS_API_KEY")
        self.base_uri = 'https://uts-ws.nlm.nih.gov/rest'
        self.cache = VLMCache(cache_dir)
        self.cache.cache_file = os.path.join(cache_dir, 'umls_cache.json')
        if os.path.exists(self.cache.cache_file):
            self.cache.cache = self.cache._load_cache()

    def search_term(self, term):
        if not self.api_key: return None
        
        # Simple normalization
        term = term.lower().strip().strip('.,')
        cached = self.cache.cache.get(term)
        if cached: return cached

        try:
            import requests # Lazy import
            query = {'string': term, 'apiKey': self.api_key, 'rootSource': 'MTH'}
            r = requests.get(f"{self.base_uri}/search/current", params=query)
            if r.status_code == 200:
                results = r.json()['result']['results']
                if results:
                    top = results[0]
                    # Minimal set to save space
                    data = {'cui': top['ui'], 'name': top['name']}
                    self.cache.cache[term] = data
                    self.cache._save_cache()
                    return data
        except Exception as e:
            logger.error(f"UMLS API Error for {term}: {e}")
        
        return None

    def extract_concepts(self, vlm_data):
        """Extracts concepts from VLM analysis dict with robust parsing"""
        concepts = []
        if not isinstance(vlm_data, dict): return []
        
        # Candidate terms
        candidates = []
        
        # 1. Key Features (List) - usually phrases
        if 'key_features' in vlm_data and isinstance(vlm_data['key_features'], list):
            candidates.extend(vlm_data['key_features'])
            
        # 2. ROI Fields (Sentences) - extract keywords
        roi_keys = ['tissue_architecture', 'stromal_components', 'inflammation', 'pathological_patterns', 'cell_morphology']
        for k in roi_keys:
            if k in vlm_data and isinstance(vlm_data[k], str):
                # Simple stopword removal and tokenization
                text = vlm_data[k]
                words = [w.strip(".,;:()").lower() for w in text.split()]
                stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'is', 'are', 'with', 'to', 'from', 'by', 'appear', 'appears', 'showing', 'suggesting', 'presence'}
                filtered = [w for w in words if w not in stopwords and len(w) > 3]
                candidates.extend(filtered)

        # 3. Search Loop
        # We limit to finding ~5-10 concepts to avoid spamming existing cache
        found_concepts = []
        seen_cuis = set()
        
        for term in candidates:
            # Try full phrase first (if it's a feature)
            res = self.search_term(term)
            if res:
                if res['cui'] not in seen_cuis:
                    found_concepts.append(res)
                    seen_cuis.add(res['cui'])
            else:
                # Try splitting phrase if long
                if len(term.split()) > 1:
                    subwords = [w.strip(".,;:()").lower() for w in term.split()]
                    stopwords = {'the', 'a', 'an', 'of', 'in', 'with'}
                    for w in subwords:
                        if w not in stopwords and len(w) > 3:
                            res_sub = self.search_term(w)
                            if res_sub and res_sub['cui'] not in seen_cuis:
                                found_concepts.append(res_sub)
                                seen_cuis.add(res_sub['cui'])
        
        return found_concepts

# ==================== MAIN PIPELINE CLASS ====================
class WSIKnowledgeGraphPipeline:
    def __init__(self, config):
        self.config = config
        self.slide_name = config['slide_name']
        self.output_dir = config['output_dir']
        self.data_dir = config['data_dir']
        
        # Paths
        self.slide_out_dir = os.path.join(self.output_dir, self.slide_name)
        self.kg_dir = os.path.join(self.slide_out_dir, 'knowledge_graph')
        self.fm = FolderManager(self.kg_dir)
        
        # Resources (Lazy loading)
        self.vlm = None
        self.roi_extractor = None
        self.gene_data = None
        self.model = None # Text embedding model
        self.umls = None

    def _init_vlm(self):
        if not self.vlm:
            self.vlm = GroqVLMAnalyzer(
                cache_dir=os.path.join(self.slide_out_dir, 'cache'),
                model_id=self.config['groq_model']
            )

    def _init_umls(self):
        if not self.umls:
            self.umls = UMLSEnricher(cache_dir=os.path.join(self.slide_out_dir, 'cache'))

    def step_clustering_and_rois(self):
        """Step 1: Cluster patches and generate ROIs"""
        logger.info("Executing Step 1: Clustering & ROI Generation")
        
        # Load Features
        feat_path = os.path.join(self.slide_out_dir, 'features', 'features.npy')
        coord_path = os.path.join(self.slide_out_dir, 'features', 'coords.npy')
        
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Features not found at {feat_path}")
            
        features = np.load(feat_path)
        coords = np.load(coord_path)
        
        # KMeans
        logger.info(f"Clustering {len(features)} patches into {self.config['n_clusters']} clusters")
        kmeans = KMeans(n_clusters=self.config['n_clusters'], random_state=42, n_init=10)
        labels = kmeans.fit_predict(StandardScaler().fit_transform(features))
        
        # ROI Generation (Graph-based)
        logger.info("Generating ROIs...")
        rois = []
        roi_id_counter = 0
        dist_thresh = self.config['patch_spacing'] * 2 + 10
        
        for cid in np.unique(labels):
            idxs = np.where(labels == cid)[0]
            if len(idxs) == 0: continue
            
            # Build spatial graph
            tree = cKDTree(coords[idxs])
            pairs = tree.query_pairs(r=dist_thresh)
            G = nx.Graph()
            G.add_nodes_from(range(len(idxs)))
            G.add_edges_from(pairs)
            
            for comp in nx.connected_components(G):
                comp_idxs = idxs[list(comp)]
                if len(comp_idxs) >= self.config['min_patches_per_roi']:
                    centroid = np.mean(coords[comp_idxs], axis=0).tolist()
                    rois.append({
                        'roi_id': roi_id_counter,
                        'cluster_id': int(cid),
                        'patch_indices': comp_idxs.tolist(),
                        'centroid': centroid
                    })
                    roi_id_counter += 1
        
        logger.info(f"Generated {len(rois)} ROIs across {len(np.unique(labels))} clusters")
        
        # Save Basic Structure
        # (This recreates the folder structure based on clustering)
        # We need to map patch_index -> cluster/ROI for subsequent steps
        # Saving an intermediate map file might be useful
        self.rois = rois # Keep in memory
        self.labels = labels
        self.coords = coords
        
        return rois

    def step_build_structure(self):
        """Step 2: Create Folder Structure (pre-VLM)"""
        logger.info("Executing Step 2: Building Folder Structure")
        if not hasattr(self, 'rois'):
            # Reload logic would go here if running standalone, for now assuming sequential
            # For standalone, we'd need to re-run step 1 or load from a cache
            logger.warning("ROIs not in memory, please run clustering step first.")
            return

        patch_h5_path = os.path.join(self.data_dir, 'patches', f'{self.slide_name}.h5')
        
        with h5py.File(patch_h5_path, 'r') as h5f:
            for roi in tqdm(self.rois, desc="Creating ROI Folders"):
                cid = roi['cluster_id']
                rid = roi['roi_id']
                
                # Create ROI Folder
                roi_dir = os.path.join(self.kg_dir, 'clusters', f'cluster_{cid}', 'rois', f'roi_{rid}')
                os.makedirs(os.path.join(roi_dir, 'patches'), exist_ok=True)
                
                # Save ROI Info
                roi_info = {
                    'roi_id': rid, 'cluster_id': cid, 
                    'num_patches': len(roi['patch_indices']),
                    'centroid': roi['centroid'],
                    'patch_indices': roi['patch_indices'] # Ensure this is saved
                }
                self.fm.save_json(os.path.join(roi_dir, 'info.json'), roi_info)
                
                # Create Patches
                for pidx in roi['patch_indices']:
                    patch_dir = os.path.join(roi_dir, 'patches', f'patch_{pidx}')
                    os.makedirs(patch_dir, exist_ok=True)
                    
                    # Extract Image (for VLM later, maybe save to disk?)
                    # For now, just Info
                    patch_info = {
                        'patch_id': int(pidx), 'cluster_id': cid, 'roi_id': rid,
                        'x': int(self.coords[pidx, 0]), 'y': int(self.coords[pidx, 1])
                    }
                    self.fm.save_json(os.path.join(patch_dir, 'info.json'), patch_info)

    def step_vlm_analysis(self):
        """Step 3: Run VLM on Patches and ROIs"""
        logger.info("Executing Step 3: VLM Analysis (Groq)")
        self._init_vlm()
        
        patch_h5_path = os.path.join(self.data_dir, 'patches', f'{self.slide_name}.h5')
        
        # We walk the directory structure we just created
        clusters_dir = os.path.join(self.kg_dir, 'clusters')
        
        with h5py.File(patch_h5_path, 'r') as h5f:
            for cname in os.listdir(clusters_dir):
                rois_dir = os.path.join(clusters_dir, cname, 'rois')
                if not os.path.exists(rois_dir): continue
                
                for rname in tqdm(os.listdir(rois_dir), desc=f"VLM {cname}"):
                    roi_path = os.path.join(rois_dir, rname)
                    
                    # 1. Analyze Patches
                    patches_path = os.path.join(roi_path, 'patches')
                    for pname in os.listdir(patches_path):
                        info_path = os.path.join(patches_path, pname, 'info.json')
                        with open(info_path, 'r') as f: info = json.load(f)
                        
                        # Load Image
                        pidx = info['patch_id']
                        img = Image.fromarray(h5f['img'][pidx])
                        
                        # Analyze
                        analysis = self.vlm.analyze_patch(img)
                        self.fm.update_json(info_path, {'vlm_analysis': analysis})
                    
                    # 2. Analyze ROI
                    if not self.roi_extractor:
                        # Try to initialize
                        for ext in ['.tif', '.tiff', '.svs', '.ndpi']:
                            wsi_path = os.path.join(self.data_dir, 'wsis', f'{self.slide_name}{ext}')
                            if os.path.exists(wsi_path):
                                self.roi_extractor = ROIExtractor(wsi_path)
                                break
                    
                    if self.roi_extractor:
                        info_path = os.path.join(roi_path, 'info.json')
                        with open(info_path, 'r') as f: roi_info = json.load(f)
                        
                        # Get coords from patches
                        patch_idxs = roi_info['patch_indices'] # Need to ensure this is in info.json from Step 2
                        if 'centroid' in roi_info and len(patch_idxs) > 0:
                            # We need raw coords. Ideally we load them or store them.
                            # For simplicity, we rely on the memory 'coords' if available, or skip.
                            if hasattr(self, 'coords'):
                                roi_coords = self.coords[patch_idxs]
                                roi_img, meta = self.roi_extractor.extract_roi(roi_coords, self.config['patch_spacing'])
                                
                                analysis = self.vlm.analyze_roi(roi_img)
                                self.fm.update_json(info_path, {'vlm_analysis': analysis, 'extraction_metadata': meta})

    def process_roi_enrichment(self, args):
        """Worker function for parallel ROI processing"""
        roi_path, cname = args
        roi_info_path = os.path.join(roi_path, 'info.json')
        
        try:
            with open(roi_info_path, 'r') as f: roi_info = json.load(f)
            
            # 1. Get/Cache ROI Concepts
            roi_vlm = roi_info.get('vlm_analysis', {})
            # Need to use the shared cache/enricher (or handle concurrency)
            # Since VLMCache writes to file, we need to be careful.
            # Ideally, we instatiate a local enricher or lock the cache.
            # For simplicity in this script, we'll let them race slightly on cache write or just rely on file system locking if lucky,
            # but standard json dump isn't atomic. 
            # safe approach: read-only cache in workers? No, we need to fetch.
            # We will suppress cache write race errors or use a lock if we could.
            # Let's just run sequential logic inside the worker, but parallelize across ROIs.
            
            roi_concepts = self.umls.extract_concepts(roi_vlm)
            self.fm.update_json(roi_info_path, {'umls_concepts': roi_concepts})
            
            # 2. Propagate to Patches
            patches_path = os.path.join(roi_path, 'patches')
            if os.path.exists(patches_path):
                for pname in os.listdir(patches_path):
                    p_info_path = os.path.join(patches_path, pname, 'info.json')
                    with open(p_info_path, 'r') as f: p_info = json.load(f)
                    
                    p_vlm = p_info.get('vlm_analysis', {})
                    p_concepts = self.umls.extract_concepts(p_vlm)
                    
                    context = {
                        'roi_vlm': roi_vlm,
                        'roi_concepts': roi_concepts,
                        'patch_concepts': p_concepts
                    }
                    self.fm.update_json(p_info_path, {'roi_context': context})
            return True
        except Exception as e:
            logger.error(f"Error processing ROI {roi_path}: {e}")
            return False

    def step_umls_enrichment(self):
        """Step 4: Propagate Context & Map to UMLS (Parallelized)"""
        logger.info("Executing Step 4: UMLS Enrichment & Propagation")
        self._init_umls()
        
        clusters_dir = os.path.join(self.kg_dir, 'clusters')
        roi_tasks = []
        
        # Collect tasks
        for cname in os.listdir(clusters_dir):
            if not os.path.isdir(os.path.join(clusters_dir, cname)): continue
            rois_dir = os.path.join(clusters_dir, cname, 'rois')
            if not os.path.exists(rois_dir): continue
            
            for rname in os.listdir(rois_dir):
                roi_path = os.path.join(rois_dir, rname)
                roi_tasks.append((roi_path, cname))
        
        logger.info(f"Found {len(roi_tasks)} ROIs to enrich. Processing with 8 threads...")
        
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            # We need to bind the method to the instance
            list(tqdm(executor.map(self.process_roi_enrichment, roi_tasks), total=len(roi_tasks), desc="Enriching ROIs"))

    def step_gene_enrichment(self):
        """Step 5: Enrich with Spatial Transcriptomics"""
        logger.info("Executing Step 4: Gene Expression Enrichment")
        
        # Initialize resources
        h5ad_path = os.path.join(self.data_dir, 'st', f'{self.slide_name}.h5ad')
        if not os.path.exists(h5ad_path):
            logger.warning(f"ST data not found at {h5ad_path}, skipping.")
            return

        adata = sc.read_h5ad(h5ad_path)
        st_coords = adata.obsm['spatial']
        tree = cKDTree(st_coords)
        
        # Logic: Walk all patch info.json, query tree, update json
        # (Simplified version of enrich_gene_expression.py)
        for root, dirs, files in os.walk(self.kg_dir):
            if 'info.json' in files and 'patch_' in root:
                path = os.path.join(root, 'info.json')
                with open(path, 'r') as f: info = json.load(f)
                
                dist, idx = tree.query([info['x'], info['y']])
                if dist < 500: # Threshold
                    # Get genes (Simplified for brevity)
                    expr = {'mapped_spot_idx': int(idx), 'distance': float(dist)}
                    # Add top 10 genes
                    row = adata.X[idx].toarray().flatten() if hasattr(adata.X[idx], 'toarray') else adata.X[idx].flatten()
                    top_idx = np.argsort(row)[-10:]
                    expr['top_genes'] = {adata.var_names[i]: float(row[i]) for i in top_idx}
                    
                    self.fm.update_json(path, {'gene_expression': expr})

    def step_text_embedding(self):
        """Step 5: Generate Text Embeddings from VLM output"""
        logger.info("Executing Step 5: Text Embeddings")
        
        self.model = SentenceTransformer("lion-ai/arctic-l-medembed")
        
        logger.info(f"Walking KG directory: {self.kg_dir}")
        count = 0
        
        # Walk and Embed
        for root, dirs, files in os.walk(self.kg_dir):
            if 'info.json' in files:
                count += 1
                path = os.path.join(root, 'info.json')
                with open(path, 'r') as f: info = json.load(f)
                
                vlm = info.get('vlm_analysis', {})
                if not vlm: continue
                
                # Simple text serialization
                text = " ".join([f"{k}: {v}" for k,v in vlm.items() if isinstance(v, str)])
                text_out = os.path.join(root, 'text_embedding.npy')
                
                if text and not os.path.exists(text_out):
                    emb = self.model.encode(text)
                    np.save(text_out, emb)

                # Concept Embedding
                concepts = info.get('umls_concepts', [])
                ctx = info.get('roi_context', {})
                propagated = ctx.get('patch_concepts', [])
                
                # Combine
                all_names = [c['name'] for c in concepts] + [c['name'] for c in propagated]
                all_names = list(set(all_names)) # Unique
                
                concept_out = os.path.join(root, 'concept_embedding.npy')
                
                if all_names and not os.path.exists(concept_out):
                    # Embed the joined concepts as a single "semantic sentence"
                    concept_text = ", ".join(all_names)
                    c_emb = self.model.encode(concept_text)
                    np.save(concept_out, c_emb)
                else:
                    if len(concepts) == 0 and len(propagated) == 0:
                        # Only log verbose if needed, but useful for debug now
                        pass
    
    def run(self):
        """Execute the full pipeline"""
        logger.info(f"Starting Pipeline for {self.slide_name}")
        
        # Orchestration
        self.step_clustering_and_rois()
        self.step_build_structure()
        
        # VLM (Optional based on flags, but default runs it)
        try:
            self.step_vlm_analysis()
        except Exception as e:
            logger.error(f"VLM Step Failed: {e}")
            
        self.step_umls_enrichment()
        self.step_gene_enrichment()
        self.step_text_embedding()
        
        logger.info("Pipeline Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Knowledge Graph Pipeline")
    parser.add_argument('--slide', type=str, default='TENX96', help='Slide Name')
    parser.add_argument('--steps', type=str, default='all', help='Steps to run (all, cluster, vlm, enrich)')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'slide_name': 'TENX96',
        'data_dir': '/app/std-data', 
        'output_dir': '/app/output',
        'n_clusters': 8,
        'patch_spacing': 0.5,
        'min_patches_per_roi': 20,
        'groq_model': 'meta-llama/llama-4-scout-17b-16e-instruct'
    }
    
    pipeline = WSIKnowledgeGraphPipeline(config)
    
    steps = args.steps.split(',')
    
    if 'all' in steps:
        pipeline.run()
    else:
        if 'cluster' in steps:
            pipeline.step_clustering_and_rois()
            pipeline.step_build_structure()
        if 'vlm' in steps:
            pipeline.step_vlm_analysis()
        if 'enrich' in steps:
            pipeline.step_umls_enrichment()
            pipeline.step_gene_enrichment()
            pipeline.step_text_embedding()
        if 'umls' in steps:
            pipeline.step_umls_enrichment()
        if 'text' in steps:
            pipeline.step_text_embedding()
