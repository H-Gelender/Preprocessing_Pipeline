# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans # Gardé pour la structure de la classe, mais non utilisé pour BanditPam
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn_extra.cluster import KMedoids # Pour la méthode BanditPam/K-Medoids
from sklearn.linear_model import LinearRegression
import torch
# kmeans_pytorch n'est pas nécessaire si on utilise KMedoids/BanditPam principalement
# from kmeans_pytorch import kmeans
import platform
import umap # Importé dans la classe originale, mais non utilisé dans les fonctions fournies
import json
import os
import shutil # Ajout pour copier les fichiers
from tqdm import tqdm # Pour afficher la progression
import re # Pour l'extraction des coordonnées des noms de fichiers (déjà présent)

class Clustering:
    """
    Classe pour effectuer le clustering, adaptée de la version fournie.
    Focalisée ici sur l'utilisation de KMedoids (BanditPam).
    """
    def __init__(self):
        self.device, self.gpu_type = self.detect_gpu()
        # L'info GPU est moins pertinente pour KMedoids de sklearn qui tourne sur CPU
        # mais on la garde pour la cohérence avec la classe originale.
        print(f"Périphérique détecté : {self.gpu_type} (Utilisé principalement par KMeans/PyTorch)")

    @staticmethod
    def detect_gpu():
        if platform.system() == 'Darwin' and torch.backends.mps.is_available():
            # Support pour Apple Silicon (Metal Performance Shaders)
            return 'mps', 'Metal (MPS)'
        elif torch.cuda.is_available():
            return 'cuda', 'CUDA (NVIDIA GPU)'
        else:
            return 'cpu', 'CPU'

    def regression(self, scores):
        """
        Méthode de régression pour trouver le "coude" (non utilisée pour BanditPam ici).
        """
        x = np.arange(len(scores))
        y = np.array(scores) # S'assurer que c'est un array numpy
        errors = []
        min_points_side = 3 # Nombre minimum de points de chaque côté du point de rupture
        k_start_offset = 2 # Les scores correspondent à k démarrant à 2

        # La plage d'indices pour les points de rupture potentiels
        # L'indice i correspond à k = i + k_start_offset
        # On a besoin d'au moins min_points_side points à gauche et à droite
        if len(x) < 2 * min_points_side:
             print("Avertissement : Pas assez de points pour la régression par morceaux.")
             # Retourner un k par défaut ou basé sur une autre heuristique?
             # Ici, on retourne le k correspondant à l'indice du milieu comme fallback simple
             return len(x) // 2 + k_start_offset

        for i in range(min_points_side, len(x) - min_points_side):
            x_left = x[:i].reshape(-1, 1)
            y_left = y[:i]
            model_left = LinearRegression().fit(x_left, y_left)
            y_left_pred = model_left.predict(x_left)
            error_left = np.sum((y_left - y_left_pred) ** 2)

            x_right = x[i:].reshape(-1, 1)
            y_right = y[i:]
            model_right = LinearRegression().fit(x_right, y_right)
            y_right_pred = model_right.predict(x_right)
            error_right = np.sum((y_right - y_right_pred) ** 2)

            errors.append(error_left + error_right)

        # L'indice du minimum dans 'errors' correspond au premier point de rupture testé (indice min_points_side dans x)
        best_split_index_in_x = np.argmin(errors) + min_points_side
        # Convertir cet indice de x en valeur k correspondante
        best_k = best_split_index_in_x + k_start_offset
        return best_k

    def find_optimal_k(self, scores, k_range, method='BanditPam'):
        """
        Trouve le nombre optimal de clusters (k) basé sur les scores d'évaluation.
        """
        if not scores:
             raise ValueError("La liste des scores est vide.")
        scores = np.array(scores) # Assure que c'est un tableau numpy

        if method == 'elbow':
            # La méthode du coude cherche le point où la diminution de l'inertie ralentit.
            # La méthode de régression tente de trouver ce point.
            optimal_k = self.regression(scores)
        elif method in ["silhouette", "calinski_harabasz", "BanditPam"]:
            # Cherche le maximum score (Silhouette, Calinski-Harabasz, ou -Inertie pour BanditPam)
            optimal_k_index = np.argmax(scores)
            optimal_k = k_range[optimal_k_index] # k correspondant à l'index max
        elif method in ["davies_bouldin", "bic"]:
             # Cherche le minimum score (Davies-Bouldin, BIC)
            optimal_k_index = np.argmin(scores)
            optimal_k = k_range[optimal_k_index] # k correspondant à l'index min
        else:
             raise ValueError(f"Méthode inconnue pour find_optimal_k: {method}")

        print(f"Basé sur la méthode '{method}', le k optimal trouvé est : {optimal_k}")
        return optimal_k

    def compute_inertia(self, X, cluster_ids_x, cluster_centers):
        """
        Calcule l'inertie pour KMeans (somme des distances au carré aux centres).
        Fonctionne avec des tenseurs PyTorch.
        """
        # S'assurer que X et cluster_centers sont des tenseurs sur le bon device
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.device)
        if isinstance(cluster_centers, np.ndarray):
            cluster_centers = torch.from_numpy(cluster_centers).to(self.device)
        if isinstance(cluster_ids_x, np.ndarray):
             cluster_ids_x = torch.from_numpy(cluster_ids_x).to(self.device)

        # S'assurer que cluster_ids_x est du bon type et sur le bon device
        cluster_ids_x = cluster_ids_x.long().to(self.device)


        dist = torch.cdist(X, cluster_centers.to(X.dtype)) # Assurer le même dtype
        # Utiliser gather pour sélectionner la distance au centre assigné pour chaque point
        min_dist_sq = dist.gather(1, cluster_ids_x.unsqueeze(1))**2
        inertia = min_dist_sq.sum().item()

        return inertia

    def run_Kmeans(self, X, num_clusters: int):
        """
        Exécute KMeans en utilisant kmeans_pytorch si GPU disponible, sinon sklearn.
        Attend un tenseur PyTorch en entrée.
        """
        print(f"KMeans exécuté sur {self.gpu_type} pour k={num_clusters}")

        # S'assurer que X est un tenseur PyTorch
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.device)
        X = X.to(self.device) # Déplacer sur le bon device

        if self.device != 'cpu' and 'kmeans_pytorch' in globals():
            # Utiliser kmeans_pytorch (si importé)
            try:
                # kmeans_pytorch peut ne pas être installé, vérifier avant d'importer
                from kmeans_pytorch import kmeans
                cluster_ids_x, cluster_centers = kmeans(
                    X=X, num_clusters=num_clusters, distance='euclidean', device=self.device, tqdm_flag=False
                )
                # Retourner tenseurs sur CPU pour cohérence ou garder sur GPU? Retournons sur CPU.
                return cluster_ids_x.cpu().numpy(), cluster_centers.cpu() # Retourner centre comme tenseur
            except ImportError:
                 print("kmeans_pytorch non trouvé. Basculement vers CPU/Sklearn.")
                 return self._run_kmeans_sklearn(X.cpu().numpy(), num_clusters)
            except Exception as e:
                 print(f"Erreur avec kmeans_pytorch : {e}. Basculement vers CPU/Sklearn.")
                 # Fallback vers CPU si kmeans_pytorch échoue
                 return self._run_kmeans_sklearn(X.cpu().numpy(), num_clusters)
        else:
            # Utiliser scikit-learn sur CPU
             return self._run_kmeans_sklearn(X.cpu().numpy(), num_clusters)

    def _run_kmeans_sklearn(self, X_np, num_clusters):
         """Helper pour exécuter KMeans avec scikit-learn."""
         print(f"Utilisation de KMeans (sklearn) sur CPU pour k={num_clusters}")
         # n_init='auto' est recommandé dans les versions récentes de sklearn
         model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto', verbose=0)
         model.fit(X_np)
         # Retourner les labels (numpy) et les centres (numpy)
         return model.labels_, model.cluster_centers_


    def evaluate_clusters(self, X, k_range=range(2, 10), method='BanditPam'):
        """
        Évalue différentes valeurs de k en utilisant la méthode spécifiée.
        Pour BanditPam, X doit être un tableau NumPy.
        Pour les autres méthodes, X peut être un tenseur ou NumPy (sera converti).
        """
        scores = []
        print(f"Évaluation des clusters pour k dans {list(k_range)} en utilisant la méthode : '{method}'")

        # S'assurer que X est NumPy pour KMedoids/BanditPam
        if method == "BanditPam":
            if isinstance(X, torch.Tensor):
                print("Conversion des données en NumPy pour KMedoids (BanditPam)...")
                X_eval = X.cpu().numpy()
            else:
                X_eval = np.asarray(X) # Assure que c'est un array numpy
        else:
             # Pour les autres méthodes, on peut travailler avec des tenseurs ou numpy
             X_eval = X


        for k in tqdm(k_range, desc=f"Évaluation {method}"):
            if method == 'elbow':
                # run_Kmeans gère la conversion Tensor/Numpy et device
                cluster_ids_x, cluster_centers = self.run_Kmeans(X_eval, k)
                # compute_inertia attend des tenseurs
                scores.append(self.compute_inertia(X_eval, cluster_ids_x, cluster_centers))

            elif method == 'silhouette':
                cluster_ids_x, _ = self.run_Kmeans(X_eval, k)
                # silhouette_score attend des numpy arrays
                if isinstance(X_eval, torch.Tensor): X_np = X_eval.cpu().numpy()
                else: X_np = np.asarray(X_eval)
                if isinstance(cluster_ids_x, torch.Tensor): cluster_ids_x_np = cluster_ids_x.cpu().numpy()
                else: cluster_ids_x_np = np.asarray(cluster_ids_x)
                # Éviter le calcul si k=1 ou nombre de labels < 2 ou > n_samples - 1
                if k < 2 or k >= X_np.shape[0]:
                     print(f"Skipping silhouette score for k={k}")
                     scores.append(-1) # Score invalide
                else:
                     try:
                         # S'assurer qu'il y a plus d'un label unique
                         if len(np.unique(cluster_ids_x_np)) > 1:
                              scores.append(silhouette_score(X_np, cluster_ids_x_np))
                         else:
                              print(f"Skipping silhouette score for k={k} (only 1 cluster found)")
                              scores.append(-1)
                     except ValueError as e:
                          print(f"Could not compute silhouette score for k={k}: {e}")
                          scores.append(-1) # Score invalide

            elif method == 'davies_bouldin':
                cluster_ids_x, _ = self.run_Kmeans(X_eval, k)
                if isinstance(X_eval, torch.Tensor): X_np = X_eval.cpu().numpy()
                else: X_np = np.asarray(X_eval)
                if isinstance(cluster_ids_x, torch.Tensor): cluster_ids_x_np = cluster_ids_x.cpu().numpy()
                else: cluster_ids_x_np = np.asarray(cluster_ids_x)
                if k < 2 or k >= X_np.shape[0]:
                     print(f"Skipping Davies-Bouldin score for k={k}")
                     scores.append(np.inf) # Score invalide (on cherche le min)
                else:
                    try:
                         if len(np.unique(cluster_ids_x_np)) > 1:
                              scores.append(davies_bouldin_score(X_np, cluster_ids_x_np))
                         else:
                              print(f"Skipping Davies-Bouldin score for k={k} (only 1 cluster found)")
                              scores.append(np.inf)
                    except ValueError as e:
                        print(f"Could not compute Davies-Bouldin score for k={k}: {e}")
                        scores.append(np.inf) # Score invalide

            elif method == 'calinski_harabasz':
                cluster_ids_x, _ = self.run_Kmeans(X_eval, k)
                if isinstance(X_eval, torch.Tensor): X_np = X_eval.cpu().numpy()
                else: X_np = np.asarray(X_eval)
                if isinstance(cluster_ids_x, torch.Tensor): cluster_ids_x_np = cluster_ids_x.cpu().numpy()
                else: cluster_ids_x_np = np.asarray(cluster_ids_x)
                if k < 2 or k >= X_np.shape[0]:
                     print(f"Skipping Calinski-Harabasz score for k={k}")
                     scores.append(-1) # Score invalide (on cherche le max)
                else:
                    try:
                         if len(np.unique(cluster_ids_x_np)) > 1:
                             scores.append(calinski_harabasz_score(X_np, cluster_ids_x_np))
                         else:
                             print(f"Skipping Calinski-Harabasz score for k={k} (only 1 cluster found)")
                             scores.append(-1)
                    except ValueError as e:
                        print(f"Could not compute Calinski-Harabasz score for k={k}: {e}")
                        scores.append(-1) # Score invalide

            elif method == "BanditPam":
                # KMedoids de sklearn_extra attend un array NumPy
                print(f"Running KMedoids (BanditPam) for k={k}...")
                # Utiliser 'k-medoids++' pour une initialisation potentiellement meilleure et plus rapide
                # 'pam' est l'algorithme classique, 'bandit' est l'approximation
                # Choisir 'pam' pour la qualité, 'bandit' pour la vitesse sur très grands datasets
                # Utilisons 'pam' ici car 'BanditPam' est dans le nom mais KMedoids l'implémente
                model = KMedoids(n_clusters=k, method='pam', init='k-medoids++', random_state=42)
                try:
                    # Vérifier si k est valide
                    if k >= X_eval.shape[0]:
                         print(f"Skipping KMedoids for k={k} (k >= n_samples)")
                         scores.append(-np.inf) # Score très mauvais
                         continue

                    model.fit(X_eval)
                    # On stocke l'inertie négative, car on veut maximiser ce score (minimiser l'inertie)
                    scores.append(-model.inertia_)
                    print(f" k={k}, -Inertia = {-model.inertia_:.4f}")
                except Exception as e:
                    print(f"Erreur lors de l'exécution de KMedoids pour k={k}: {e}")
                    scores.append(-np.inf) # Score très mauvais en cas d'erreur

            else:
                raise ValueError("Unknown method, choose from : 'elbow', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'BanditPam'")
        return scores

    def apply_clustering(self, X, k: int, method: str):
        """
        Applique l'algorithme de clustering final avec le k choisi.
        Pour BanditPam, X doit être NumPy.
        """
        print(f"\nApplication du clustering final avec k={k} en utilisant la méthode '{method}'...")
        # S'assurer que k est un entier
        k_int = int(k)
        if k_int != k:
             print(f"Avertissement: k={k} a été converti en entier {k_int}")
        k = k_int

        if k < 1:
            raise ValueError(f"Le nombre de clusters k doit être au moins 1, mais reçu {k}")


        if method == "BanditPam":
            # KMedoids attend NumPy
            if isinstance(X, torch.Tensor):
                X_final = X.cpu().numpy()
            else:
                X_final = np.asarray(X)

            if k >= X_final.shape[0]:
                 raise ValueError(f"Le nombre de clusters k={k} ne peut pas être >= au nombre d'échantillons n={X_final.shape[0]}")


            print(f"Exécution de KMedoids (method='pam', init='k-medoids++') avec k={k}...")
            model = KMedoids(n_clusters=k, method='pam', init='k-medoids++', random_state=42)
            try:
                model.fit(X_final)
                # Retourne les labels et les centres (qui sont des points de données réels)
                cluster_ids_x = model.labels_
                cluster_centers = model.cluster_centers_
                print("Clustering KMedoids terminé.")
            except Exception as e:
                 print(f"Erreur lors de l'application finale de KMedoids : {e}")
                 raise e # Propage l'erreur
        else:
            # Utiliser run_Kmeans pour les autres méthodes (qui gère Tensor/Numpy/Device)
            cluster_ids_x, cluster_centers = self.run_Kmeans(X, k)
            print(f"Clustering {method} (via KMeans) terminé.")

        return cluster_ids_x, cluster_centers

    def extract_labels(self, cluster_ids_x, pos, file_name='cords_labels.json'):
        """
        Crée un dictionnaire mappant les ID de cluster aux coordonnées correspondantes
        et le sauvegarde dans un fichier JSON.

        Args:
            cluster_ids_x (np.ndarray): Array contenant l'ID de cluster pour chaque point.
            pos (np.ndarray): Array des coordonnées [x, y] correspondant à chaque point.
            file_name (str): Nom du fichier JSON de sortie.

        Returns:
            dict: Le dictionnaire créé.
        """
        # S'assurer que les labels sont des entiers natifs Python pour la sérialisation JSON
        cluster_ids_native = [int(label) for label in cluster_ids_x]
        unique_labels = sorted(list(set(cluster_ids_native)))

        dict_classification = {label: [] for label in unique_labels}

        if len(cluster_ids_native) != len(pos):
            raise ValueError(f"Le nombre de labels ({len(cluster_ids_native)}) ne correspond pas au nombre de positions ({len(pos)})")

        print(f"Association des {len(pos)} coordonnées aux {len(unique_labels)} clusters...")
        for i in range(len(pos)):
            label = cluster_ids_native[i]
            # Convertir les coordonnées numpy en liste de int/float natifs Python
            coordinate = [int(c) for c in pos[i]] # Assurer que ce sont des entiers si attendu
            dict_classification[label].append(coordinate)

        print(f"Sauvegarde de la classification dans : {file_name}")
        try:
            # Créer le dossier parent si nécessaire
            output_dir = os.path.dirname(file_name)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir)

            with open(file_name, 'w') as file:
                json.dump(dict_classification, file, indent=4)
            print(f"Fichier JSON '{file_name}' créé avec succès !")
        except IOError as e:
            print(f"Erreur lors de l'écriture du fichier JSON {file_name}: {e}")
        except TypeError as e:
             print(f"Erreur de type lors de la sérialisation JSON (vérifiez les types de données) : {e}")


        return dict_classification

# --- Nouvelle fonction pour organiser les patches ---
def organize_patches_by_cluster(
    cluster_labels: np.ndarray,
    positions: np.ndarray,
    patch_source_folder: str,
    clustered_output_folder: str,
    patch_size: int,
    patch_level: int = 0 # Niveau auquel les patches ont été extraits (pour le nom de fichier)
    ):
    """
    Copie les fichiers image des patches dans des sous-dossiers basés sur leur cluster.

    Args:
        cluster_labels (np.ndarray): Array contenant l'ID de cluster pour chaque patch.
        positions (np.ndarray): Array des coordonnées [x, y] (niveau 0) pour chaque patch.
        patch_source_folder (str): Dossier où se trouvent les images originales des patches.
        clustered_output_folder (str): Dossier racine où créer les sous-dossiers par cluster.
        patch_size (int): Taille (largeur/hauteur) des patches (utilisée pour reconstruire le nom).
        patch_level (int): Niveau d'extraction des patches (utilisé pour reconstruire le nom).
    """
    print(f"\nOrganisation des patches par cluster dans : {clustered_output_folder}")
    if not os.path.isdir(patch_source_folder):
        print(f"Erreur : Le dossier source des patches '{patch_source_folder}' n'existe pas.")
        return

    if len(cluster_labels) != len(positions):
        print("Erreur : Le nombre de labels ne correspond pas au nombre de positions.")
        return

    copied_count = 0
    skipped_count = 0
    # Créer le dossier de sortie principal s'il n'existe pas
    os.makedirs(clustered_output_folder, exist_ok=True)

    unique_labels = sorted(list(set(cluster_labels)))

    # Créer les sous-dossiers pour chaque cluster
    for label in unique_labels:
        cluster_dir = os.path.join(clustered_output_folder, f"cluster_{label}")
        os.makedirs(cluster_dir, exist_ok=True)

    # Itérer sur chaque patch, reconstruire son nom et le copier
    print("Copie des fichiers patches dans les dossiers de cluster...")
    for i in tqdm(range(len(positions)), desc="Copie des patches"):
        label = cluster_labels[i]
        x, y = positions[i]

        # Reconstruire le nom de fichier attendu
        # Format: patch_L{level}_x{X}_y{Y}_s{size}.png (basé sur patch_generator_func_fr)
        patch_filename = f"patch_L{patch_level}_x{x}_y{y}_s{patch_size}.png"
        source_path = os.path.join(patch_source_folder, patch_filename)

        # Définir le chemin de destination
        destination_dir = os.path.join(clustered_output_folder, f"cluster_{label}")
        destination_path = os.path.join(destination_dir, patch_filename)

        # Vérifier si le fichier source existe et copier
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, destination_path) # copy2 préserve les métadonnées
                copied_count += 1
            except Exception as e:
                print(f"\nErreur lors de la copie de {patch_filename} vers {destination_dir}: {e}")
                skipped_count += 1
        else:
            # print(f"\nAvertissement : Fichier patch source non trouvé : {source_path}")
            skipped_count += 1

    print(f"\nOrganisation terminée.")
    print(f"  {copied_count} patches copiés.")
    if skipped_count > 0:
        print(f"  {skipped_count} patches sources non trouvés ou erreurs de copie.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Paramètres ---
    POSITIONS_FILE = "extracted_positions.npy"
    FEATURES_FILE = "extracted_features.npy"
    OUTPUT_JSON_FILE = "clustered_coordinates.json"
    PATCH_SOURCE_FOLDER = "output_patches" # Dossier contenant les .png des patches
    CLUSTERED_PATCHES_OUTPUT_FOLDER = "clustered_patches" # Dossier où organiser les patches par cluster
    PATCH_SIZE = 224 # IMPORTANT: Doit correspondre à la taille utilisée dans patch_generator_func_fr
    PATCH_LEVEL = 0  # Niveau d'extraction utilisé dans patch_generator_func_fr

    CLUSTERING_METHOD = "BanditPam" # Méthode à utiliser (KMedoids)
    K_RANGE = range(2, 11) # Plage de k à tester (de 2 à 10 clusters)

    # --- Chargement des données ---
    print("Chargement des données...")
    try:
        positions = np.load(POSITIONS_FILE)
        features = np.load(FEATURES_FILE)
        print(f"Positions chargées : shape={positions.shape}, dtype={positions.dtype}")
        print(f"Caractéristiques chargées : shape={features.shape}, dtype={features.dtype}")

        if features.shape[0] != positions.shape[0]:
             raise ValueError("Le nombre de points dans les fichiers de positions et de caractéristiques ne correspond pas.")
        if features.shape[0] == 0:
             print("Erreur: Aucun point trouvé dans les fichiers chargés.")
             exit()
        # S'assurer que les features sont en float32 pour KMedoids (peut éviter des problèmes)
        if features.dtype != np.float32:
             print(f"Conversion des caractéristiques de {features.dtype} vers float32.")
             features = features.astype(np.float32)


    except FileNotFoundError as e:
        print(f"Erreur : Fichier d'entrée non trouvé : {e}")
        print(f"Assurez-vous que les fichiers '{POSITIONS_FILE}' et '{FEATURES_FILE}' existent.")
        exit(1)
    except Exception as e:
        print(f"Erreur lors du chargement des fichiers .npy : {e}")
        exit(1)

    # --- Initialisation du Clustering ---
    try:
        clusterer = Clustering()
    except Exception as e:
        print(f"Erreur lors de l'initialisation de la classe Clustering : {e}")
        exit(1)

    # --- Évaluation pour trouver k optimal ---
    try:
        # Utiliser KMedoids (BanditPam) pour l'évaluation
        # La méthode evaluate_clusters pour BanditPam retourne -Inertie
        evaluation_scores = clusterer.evaluate_clusters(features, k_range=K_RANGE, method=CLUSTERING_METHOD)

        if not evaluation_scores or np.all(np.isinf(evaluation_scores)): # Vérifier si tous les scores sont infinis (erreur)
             print("Erreur : Aucun score d'évaluation valide n'a été retourné. Vérifiez la plage de k et les données.")
             exit(1)

        # Trouver k optimal en maximisant -Inertie (donc en maximisant le score retourné)
        optimal_k = clusterer.find_optimal_k(evaluation_scores, k_range=K_RANGE, method=CLUSTERING_METHOD)

    except Exception as e:
        print(f"Erreur lors de l'évaluation des clusters ou de la recherche de k optimal : {e}")
        exit(1)

    # --- Application du Clustering Final ---
    try:
        # Appliquer KMedoids (BanditPam) avec le k optimal
        final_labels, final_centers = clusterer.apply_clustering(features, k=optimal_k, method=CLUSTERING_METHOD)

        if final_labels is None:
             print("Erreur : Le clustering final n'a pas retourné de labels.")
             exit(1)

    except Exception as e:
        print(f"Erreur lors de l'application du clustering final : {e}")
        exit(1)

    # --- Extraction et Sauvegarde des Labels (JSON) ---
    try:
        classification_dict = clusterer.extract_labels(final_labels, positions, file_name=OUTPUT_JSON_FILE)

    except Exception as e:
        print(f"Erreur lors de l'extraction ou de la sauvegarde des labels JSON : {e}")
        # Ne pas quitter forcément, on peut encore essayer d'organiser les patches
        # exit(1)

    # --- Organisation des Patches dans des Dossiers par Cluster ---
    try:
        organize_patches_by_cluster(
            cluster_labels=final_labels,
            positions=positions,
            patch_source_folder=PATCH_SOURCE_FOLDER,
            clustered_output_folder=CLUSTERED_PATCHES_OUTPUT_FOLDER,
            patch_size=PATCH_SIZE,
            patch_level=PATCH_LEVEL
        )
    except Exception as e:
         print(f"Erreur lors de l'organisation des patches par cluster : {e}")
         exit(1)


    print("\nScript de clustering et d'organisation terminé avec succès.")