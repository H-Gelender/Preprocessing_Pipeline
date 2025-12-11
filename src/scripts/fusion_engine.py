import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from typing import Tuple, Dict

class MultiModalFusion:
    """
    Performs Joint Fusion of Image Embeddings and Reduced CUI Concepts.
    """
    def __init__(self, n_clusters: int = 8, svd_components: int = 50):
        self.n_clusters = n_clusters
        self.svd_components = svd_components
        
        # State
        self.vectorizer_cui = None
        self.scaler = StandardScaler()
        self.kmeans = None
        
    def process_fusion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the full fusion pipeline:
        1. TF-IDF on CUIs
        2. SVD Reduction
        3. Concat with Image Embeddings
        4. Clustering
        """
        print("--- Starting Multi-Modal Fusion ---")
        
        # 1. TF-IDF (Bag of CUIs)
        print("Vectorizing CUIs...")
        docs_cui = df['cuis'].apply(lambda x: " ".join(x))
        # Min 5 occurrences to be significant
        self.vectorizer_cui = TfidfVectorizer(min_df=5, max_df=0.85, lowercase=False)
        X_tfidf = self.vectorizer_cui.fit_transform(docs_cui)
        print(f"Vocabulary size: {len(self.vectorizer_cui.get_feature_names_out())} unique CUIs")
        
        # 2. SVD Reduction
        print(f"Reducing dimensions (SVD={self.svd_components})...")
        svd = TruncatedSVD(n_components=self.svd_components, random_state=42)
        X_cui_red = svd.fit_transform(X_tfidf)
        
        # 3. Normalization & Fusion
        print("Normalizing and Fusing...")
        X_img = np.vstack(df['image_embedding'].values)
        
        X_img_norm = self.scaler.fit_transform(X_img)
        X_cui_norm = self.scaler.fit_transform(X_cui_red)
        
        X_fusion = np.hstack([X_img_norm, X_cui_norm])
        
        # 4. Clustering
        print(f"Clustering into {self.n_clusters} Multi-Modal Domains...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        df['joint_cluster'] = self.kmeans.fit_predict(X_fusion)
        df['joint_cluster'] = df['joint_cluster'].astype(str)
        
        # 5. Profiling (Interpretability)
        self._assign_domain_descriptions(df, X_tfidf)
        
        # 6. Evaluation
        if 'global_cluster_id' in df.columns:
            score = adjusted_rand_score(df['global_cluster_id'], df['joint_cluster'])
            print(f"Agreement with Spatial Clusters (ARI): {score:.4f}")
            
        return df, X_fusion

    def _assign_domain_descriptions(self, df: pd.DataFrame, X_tfidf):
        """Identifies top concepts for each cluster and adds to DF."""
        feature_names = np.array(self.vectorizer_cui.get_feature_names_out())
        
        # Global Map CUI -> Name
        global_map = {}
        for m in df['cui_map']:
            global_map.update(m)
            
        print("\n--- Domain Concept Profiles ---")
        for cid in sorted(df['joint_cluster'].unique()):
            mask = df['joint_cluster'] == cid
            
            # Mean TF-IDF vector for this cluster
            centroid = X_tfidf[mask.values].mean(axis=0).A1
            
            # Top 5 concepts
            top_idx = centroid.argsort()[-5:][::-1]
            names = [global_map.get(feature_names[i], feature_names[i]) for i in top_idx]
            
            desc = ", ".join(names)
            print(f"Domain {cid}: {desc}")
            
            df.loc[mask, 'domain_desc'] = desc

    def analyze_profile(self, df: pd.DataFrame, top_n=10, rare_n=5):
        """Prints detailed Top (Dominant) and Bottom (Rare) concepts per cluster."""
        print("\n=== Detailed Cluster Analysis ===")
        
        feature_names = np.array(self.vectorizer_cui.get_feature_names_out())
        # Global Frequency (how many docs contain this CUI)
        # We can approximate this from vectorizer.idf_:  df = (N+1) / exp(idf - 1) - 1
        # Or just use the stop words info if available, but easier to re-calc or use idf rank.
        # Higher IDF = Rarer.
        global_idf = self.vectorizer_cui.idf_
        
        # Map CUI -> Readable Name
        global_map = {}
        for m in df['cui_map']: global_map.update(m)
        
        X_tfidf = self.vectorizer_cui.transform(df['cuis'].apply(lambda x: " ".join(x)))

        for cid in sorted(df['joint_cluster'].unique()):
            mask = df['joint_cluster'] == cid
            subset_size = mask.sum()
            print(f"\n--- Cluster {cid} (n={subset_size} patches) ---")
            
            # 1. Top Prominent (Highest TF-IDF in Centroid)
            # Use X_tfidf[mask.values] to avoid Series indexing error
            centroid = X_tfidf[mask.values].mean(axis=0).A1 
            top_indices = centroid.argsort()[-top_n:][::-1]
            
            print(f"[{'Top ' + str(top_n) + ' Dominant':^30}]")
            for idx in top_indices:
                if centroid[idx] > 0:
                    cui = feature_names[idx]
                    name = global_map.get(cui, cui)
                    print(f"  {name[:40]:<40} (Score: {centroid[idx]:.3f})")

            # 2. Rarest (High IDF that are present in this cluster)
            # We only care about concepts that actually appear in this cluster (centroid > 0)
            active_indices = np.where(centroid > 0)[0]
            if len(active_indices) > 0:
                # Sort active indices by IDF (Descending = Rarer)
                rare_indices = active_indices[np.argsort(global_idf[active_indices])[::-1]][:rare_n]
                
                print(f"[{'Top ' + str(rare_n) + ' Rarest/Specific':^30}]")
                for idx in rare_indices:
                    cui = feature_names[idx]
                    name = global_map.get(cui, cui)
                    idf_score = global_idf[idx]
                    print(f"  {name[:40]:<40} (Rarity: {idf_score:.2f})")

    def get_top_concepts_df(self, df: pd.DataFrame, top_n=10) -> pd.DataFrame:
        """Returns a DataFrame of top concepts per cluster for plotting."""
        records = []
        feature_names = np.array(self.vectorizer_cui.get_feature_names_out())
        global_map = {}
        for m in df['cui_map']: global_map.update(m)
        
        X_tfidf = self.vectorizer_cui.transform(df['cuis'].apply(lambda x: " ".join(x)))
        
        for cid in sorted(df['joint_cluster'].unique()):
            mask = df['joint_cluster'] == cid
            centroid = X_tfidf[mask.values].mean(axis=0).A1 
            top_indices = centroid.argsort()[-top_n:][::-1]
            
            for rank, idx in enumerate(top_indices):
                score = centroid[idx]
                if score > 0:
                    cui = feature_names[idx]
                    name = global_map.get(cui, cui)
                    records.append({
                        'Cluster': f"C{cid}",
                        'Concept': name,
                        'Score': score,
                        'Rank': rank + 1
                    })
        return pd.DataFrame(records)

    def compute_cluster_similarity(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Computes pairwise cosine similarity between cluster centroids for CUI and Image modalities."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        clusters = sorted(df['joint_cluster'].unique())
        n_clusters = len(clusters)
        
        # 1. CUI Centroids
        X_tfidf = self.vectorizer_cui.transform(df['cuis'].apply(lambda x: " ".join(x)))
        cui_centroids = []
        for cid in clusters:
            mask = df['joint_cluster'] == cid
            # Sparse mean returns np.matrix, convert to array
            centroid = np.asarray(X_tfidf[mask.values].mean(axis=0))
            cui_centroids.append(centroid.flatten())
        
        cui_centroids = np.vstack(cui_centroids)
        sim_cui = cosine_similarity(cui_centroids)
        
        # 2. Image Centroids
        X_img = np.vstack(df['image_embedding'].values)
        img_centroids = []
        for cid in clusters:
            mask = df['joint_cluster'] == cid
            img_centroids.append(X_img[mask.values].mean(axis=0))
        img_centroids = np.vstack(img_centroids)
        sim_img = cosine_similarity(img_centroids)
        
        # Wrap in DataFrames
        df_cui = pd.DataFrame(sim_cui, index=clusters, columns=clusters)
        df_img = pd.DataFrame(sim_img, index=clusters, columns=clusters)
        
        return df_cui, df_img

    def process_gene_analysis(self, df: pd.DataFrame, n_clusters=8, marked_genes: list = None) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Clustering and Profiling on Gene Expression."""
        print("\n--- Starting Gene Expression Analysis ---")
        
        # 1. Extract Gene Matrix
        gene_cols = [c for c in df.columns if c.startswith('tgt_')]
        if not gene_cols:
            print("No gene expression data found (tgt_*)")
            return df, None, None, None
            
        print(f"Found {len(gene_cols)} genes.")
        X_gene = df[gene_cols].fillna(0).values
        
        # 2. Normalize (Log1p + Scale) - standard for expression
        X_gene_log = np.log1p(X_gene)
        X_gene_norm = self.scaler.fit_transform(X_gene_log)
        
        # 3. Clustering
        print(f"Clustering Genes into {n_clusters} spatial domains...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_gene_norm)
        df['gene_cluster'] = kmeans.predict(X_gene_norm).astype(str)
        
        # 4. Profiling (Top Genes)
        print("Analying Top Genes per Cluster...")
        # Clean col names (remove tgt_)
        gene_names = np.array([g.replace('tgt_', '') for g in gene_cols])
        name_to_idx = {name: i for i, name in enumerate(gene_names)}
        
        records = []
        marked_records = []
        
        for cid in sorted(df['gene_cluster'].unique()):
            mask = df['gene_cluster'] == cid
            centroid = X_gene[mask].mean(axis=0) # Use raw expression for meaning
            
            # A. Top Expressed
            top_idx = centroid.argsort()[-10:][::-1]
            for rank, idx in enumerate(top_idx):
                if centroid[idx] > 0:
                    records.append({
                        'Cluster': f"G{cid}",
                        'Gene': gene_names[idx],
                        'Expression': centroid[idx],
                        'Rank': rank + 1
                    })
            
            # B. Marked Genes (if provided)
            if marked_genes:
                for gene in marked_genes:
                    if gene in name_to_idx:
                        idx = name_to_idx[gene]
                        marked_records.append({
                            'Cluster': f"G{cid}",
                            'Gene': gene,
                            'Expression': centroid[idx]
                        })

        self.gene_profile_df = pd.DataFrame(records)
        self.marked_profile_df = pd.DataFrame(marked_records)
        
        return df, X_gene_norm, self.gene_profile_df, self.marked_profile_df

    def compute_all_similarities(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Computes similarity matrices for CUI, Image, and Gene modalities."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        clusters = sorted(df['joint_cluster'].unique())
        results = {}
        
        # Helper
        def get_sim(X_data):
            centroids = []
            for cid in clusters:
                mask = df['joint_cluster'] == cid
                centroids.append(np.asarray(X_data[mask.values].mean(axis=0)).flatten())
            return pd.DataFrame(cosine_similarity(np.vstack(centroids)), index=clusters, columns=clusters)

        # 1. CUI
        X_tfidf = self.vectorizer_cui.transform(df['cuis'].apply(lambda x: " ".join(x)))
        results['cui'] = get_sim(X_tfidf)
        
        # 2. Image
        X_img = np.vstack(df['image_embedding'].values)
        results['img'] = get_sim(X_img)
        
        # 3. Gene (if available)
        gene_cols = [c for c in df.columns if c.startswith('tgt_')]
        if gene_cols:
            X_gene = np.log1p(df[gene_cols].fillna(0).values)
            results['gene'] = get_sim(X_gene)
            
        if gene_cols:
            X_gene = np.log1p(df[gene_cols].fillna(0).values)
            results['gene'] = get_sim(X_gene)
            
        return results

    def compute_cluster_overlap(self, df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
        """Computes the confusion matrix between two cluster assignments."""
        # Check columns exist
        if col1 not in df.columns or col2 not in df.columns:
            print(f"Columns {col1} or {col2} not found in DataFrame.")
            return pd.DataFrame()
            
        if col1 not in df.columns or col2 not in df.columns:
            print(f"Columns {col1} or {col2} not found in DataFrame.")
            return pd.DataFrame()
            
        # Cross Tabulation
        ct = pd.crosstab(df[col1], df[col2])
        return ct

    def compute_unimodal_clusters(self, df: pd.DataFrame, n_clusters=8) -> pd.DataFrame:
        """Generates independent clusters for Image and CUI modalities."""
        print(f"Computing Unimodal Clusters (k={n_clusters})...")
        
        # 1. Image
        X_img = np.vstack(df['image_embedding'].values)
        kmeans_img = KMeans(n_clusters=n_clusters, random_state=42).fit(X_img)
        df['img_cluster'] = kmeans_img.labels_.astype(str)
        
        # 2. CUI
        # Re-vectorize if needed or use existing if cached (not cached in self, so re-run)
        docs_cui = df['cuis'].apply(lambda x: " ".join(x))
        X_tfidf = self.vectorizer_cui.transform(docs_cui)
        svd = TruncatedSVD(n_components=self.svd_components, random_state=42)
        X_cui_red = svd.fit_transform(X_tfidf)
        
        kmeans_cui = KMeans(n_clusters=n_clusters, random_state=42).fit(X_cui_red)
        df['cui_cluster'] = kmeans_cui.labels_.astype(str)
        
        return df

    def compute_marker_discrimination(self, df: pd.DataFrame, gene_col: str, cluster_col: str) -> float:
        """Calculates ANOVA F-value quite safely."""
        from sklearn.feature_selection import f_classif
        import warnings
        
        if gene_col not in df.columns or cluster_col not in df.columns:
            return 0.0
            
        # Drop Na
        data = df[[gene_col, cluster_col]].dropna()
        if len(data) < 2: return 0.0
        
        X = data[[gene_col]]
        y = data[cluster_col]
        
        # Check variance (if gene is all 0, F is undefined)
        if X[gene_col].var() == 0:
            return 0.0
            
        # F-stats with warning suppression
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f, p = f_classif(X, y)
            return float(f[0]) if not np.isnan(f[0]) else 0.0
        except:
            return 0.0
