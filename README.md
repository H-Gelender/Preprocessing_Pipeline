# Multi-Modal Spatial Analysis Pipeline

## Project Overview
This project implements a comprehensive pipeline for analyzing spatial transcriptomics and pathology data. It integrates visual features (from histology images), semantic concepts (CUI from medical knowledge graphs), and gene expression data to identify and characterize spatial domains.

## Capabilities & Methodology

### 1. Feature Extraction
*   **Visual Embeddings**: 
    *   **Source**: Histology patches (224x224 px).
    *   **Method**: ResNet50 (Pretrained on ImageNet), utilizing the penultimate layer to generate 2048-dimensional embeddings.
    *   **Script**: `src/scripts/generate_image_embeddings.py`
    *   **Output**: stored as `image_embedding.npy` per patch.
    
*   **Concept Extraction (ROIs)**: 
    *   **Source**: Regions of Interest identified in the slide.
    *   **Method**: Mapped to CUI (Concept Unique Identifiers) via Knowledge Graph.
    *   **Processing**: TF-IDF vectorization followed by SVD (Singular Value Decomposition) for dimensionality reduction (default 50 components).

### 2. Multi-Modal Fusion & Clustering
*   **Fusion Engine**:
    *   **Method**: Concatenation of normalized Image Embeddings and Reduced CUI Vectors.
    *   **Clustering**: K-Means clustering (default k=8) on the joint fusion space.
    *   **Validation**: Adjusted Rand Index (ARI) to compare against spatial ground truth or other modalities.
    *   **Script**: `src/scripts/fusion_engine.py`

### 3. Dimensionality Reduction & Visualization
*   **Projection**: UMAP for 2D visualization of the high-dimensional fusion space.
*   **Outputs**: Heatmaps of cluster similarity, spatial cluster maps, and gene expression overlays.

### 4. Gene Expression Analysis
*   **Target**: `tgt_*` columns (Gene Expression).
*   **Processing**: Log1p normalization + Standardization.
*   **Analysis**: Clustering of gene expression profiles to identify "Gene Domains".
*   **Profiling**: Identification of top expressed genes and specific marker genes (e.g., *ESR1*, *KRT8*) per cluster.

---

## Current Status (TENX96)

We have successfully processed the **TENX96** slide. Below are the statistics of the extracted elements:

*   **Patches Extracted**: 5,124
*   **ROIs Identified**: 42
*   **Feature Vectors**: 6,177 (Total feature instances processed)
*   **Cluster Analysis**:
    *   Multi-modal clusters generated (Image + CUI).
    *   Gene expression clusters generated.
    *   Similarity matrices computed between Image, CUI, and Gene clusters.

---

# 🚀 Plan d'Exécution Technique : Système Neuro-Symbolique WSI

*Ce document sert de spécification technique pour l'ingénieur en charge de l'implémentation. Chaque tâche est atomique et vérifiable.*

## 📅 Phase 1 : Pipeline de Prétraitement & Vectorisation
*(Sursis Estimé: 2 semaines)*

**Objectif Final** : Convertir des fichiers `.svs` / `.ndpi` (images gigapixels) en un ensemble de fichiers `.pt` (PyTorch tensors) structurés géographiquement et sémantiquement.

### Tâche 1.1 : Segmentation & Patching (CLAM Framework)
*   **Action** : Configurer le framework CLAM (ou équivalent) pour traiter les WSI.
*   **Spécifications** :
    *   **Niveau de Zoom** : 20x (0.5 MPP) ou 40x selon la résolution de la Transcriptomique Spatiale (ST).
    *   **Taille de patch** : Doit correspondre à la taille du spot ST (ex: $224 \times 224$ px pour Visium).
    *   **Alignement ST (CRITIQUE)** : Pour chaque spot de transcriptomique $(x, y)$, identifier le patch image correspondant exact.
*   **Output Technique** :
    *   Fichier `st_mapping.csv` : Colonnes `patch_id`, `x_coord`, `y_coord`, `gene_counts_vector`.

### Tâche 1.2 : Entraînement/Fine-tuning de l'Encodeur CUI
*   **Action** : Créer le "Vocabulaire Visuel".
*   **Architecture** : ResNet50 ou H-optimus-0.
*   **Output Technique** :
    *   `cui_encoder.pth` et fonction `get_cui_probs(patch)`.

### Tâche 1.3 : Construction Hiérarchique (Patch $\rightarrow$ ROI $\rightarrow$ Cluster)
*   **Action** : Préparer les structures de données pour le chargement dynamique.
*   **Structure de Données (HDF5 hiérarchique)** :
    *   Groupe `/clusters/` : Liste des Clusters et leurs vecteurs moyens.
    *   Groupe `/rois/` : Liste des ROIs, lien vers Cluster parent, et lien vers liste des Patchs enfants.
    *   Groupe `/patches/` : Les features individuelles des patchs (ne sont chargées que sur demande).
*   **Validation Phase 1** : Vérifier qu'on peut récupérer instantanément les features des 50 patchs d'un ROI donné sans charger toute la lame.

---

## 🧠 Phase 2 : Baseline GNN Multi-Échelle
*(Sursis Estimé: 1 semaine)*

**Objectif Final** : Valider la capacité prédictive à haute résolution.

### Tâche 2.1 : Graphe Pyramidal
*   **Action** : Construire un graphe statique pour le test.
*   **Architecture** :
    *   Utiliser un **Hierarchical Graph Pooling** (ex: HGP-SL ou DiffPool).
    *   **Niveau 1** : GNN sur les Patchs $\rightarrow$ Embedding ROI.
    *   **Niveau 2** : GNN sur les ROIs $\rightarrow$ Embedding Cluster.
*   **Output Technique** : Script `train_hierarchical_baseline.py`.

---

## 🗄️ Phase 3 : Backend Knowledge Graph (SQLite)
*(Sursis Estimé: 1 semaine)*

**Objectif Final** : Persistance des connaissances.

### Tâche 3.1 à 3.3
*(Identique à la version précédente)*
Le Knowledge Graph stocke les concepts et les gènes. Il sert de "mémoire sémantique" pour guider l'agent, quel que soit son niveau de zoom.

---

## 🕵️ Phase 4 : Agent RL Navigateur & Zoom
*(Sursis Estimé: 3 semaines)*

**Objectif Final** : Un agent capable de naviguer et de zoomer jusqu'au patch précis.

### Tâche 4.1 : Espace d'État Hiérarchique
*   **Action** : L'observation dépend du niveau de zoom ($L$).
*   **Vecteur d'État** :
    $$S_t = [ \text{Node\_Feat}, \text{Global\_Context}, \text{Level\_Indicator} ]$$
    *   Si $L=0$ (Cluster) : `Node_Feat` = Moyenne du Cluster.
    *   Si $L=1$ (ROI) : `Node_Feat` = Moyenne du ROI.
    *   Si $L=2$ (Patch) : `Node_Feat` = Features précises du Patch (CUI vector).

### Tâche 4.2 : Espace d'Action Dynamique (Zoom)
*   **Action** : Implémenter la logique de transition de couche.
*   **Actions Possibles** :
    *   **Move** : Aller au voisin (même niveau).
    *   **Zoom In** : Descendre dans le nœud actuel (charger le sous-graphe). *Action disponible seulement si niveau < 2.*
    *   **Predict/Stop** : Faire la prédiction sur le nœud actuel (*disponible seulement si niveau = 2 / Patch*).

### Tâche 4.3 : Environnement "Lazy Loading" (Gym)
*   **Action** : Coder la méthode `step()` pour gérer le chargement de données.
*   **Logique** :
    ```python
    def step(self, action):
        if action == ZOOM_IN:
            self.current_level += 1
            # CHARGEMENT DYNAMIQUE ICI
            # On récupère les IDs des enfants du noeud courant depuis le H5
            # On construit le petit graphe (50-100 noeuds) à la volée
            self.subgraph = self.load_children(self.current_node)
            return self.get_observation(self.subgraph.start_node)
    ```
*   **Validation Phase 4** : L'agent doit partir de la vue globale et finir sur un patch précis de nécrose pour prédire une hypoxie locale, en moins de 20 "pas" au total.

---

## 🔭 Phase 5 : Visualisation Zoomable
*(Sursis Estimé: 1 semaine)*

**Objectif Final** : Interface Web permettant de suivre l'agent dans les profondeurs.

### Tâche 5.2 : Frontend Dash Cytoscape
*   **Action** : Mettre à jour l'interface.
*   **Feature** : "Drill-down".
    *   Clic sur un nœud Cluster $\rightarrow$ Affiche le graphe des ROIs.
    *   Clic sur un nœud ROI $\rightarrow$ Affiche le graphe des Patchs (images réelles).

---

## ♻️ Phase 6 : Boucle d'Optimisation
*(Sursis Estimé: 2 semaines)*

**Objectif Final** : Apprentissage continu.

### Tâche 6.1 : Prioritized Experience Replay (PER)
*   **Conservation des Trajectoires** : Une trajectoire est maintenant une séquence hiérarchique :
    `Cluster_A -> Zoom -> ROI_42 -> Zoom -> Patch_12 -> Predict`.