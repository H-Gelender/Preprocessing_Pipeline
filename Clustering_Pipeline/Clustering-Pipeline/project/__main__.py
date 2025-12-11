# -*- coding: utf-8 -*-
import os
import argparse
import time
import numpy as np
import torch
import cv2 # Pour sauvegarder le masque
import json # Pour charger la configuration JSON
import traceback # Importer pour afficher la trace de l'erreur

# --- Importer les fonctions depuis les autres modules ---
# Assurez-vous que ces fichiers sont dans le même dossier ou dans le PYTHONPATH
try:
    from mask_extractor import generate_tissue_mask
    from tissue_extractor import generate_tissue_patches
    from feature_extractor import setup_model_and_transform, extract_features
    from clustering import Clustering, organize_patches_by_cluster
    from vis_clusters import visualize_clusters
except ImportError as e:
    print(f"Erreur d'importation : {e}")
    print("Assurez-vous que les fichiers .py (tissue_masker.py, patch_generator.py, etc.) sont présents.")
    exit(1)

def load_config_from_json(config_path: str) -> dict:
    """
    Charge la configuration depuis un fichier JSON spécifié.
    """
    print(f"Chargement de la configuration depuis : {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier de configuration '{config_path}' n'a pas été trouvé.")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            print("  Fichier ouvert avec succès. Tentative de chargement JSON...")
            config_data = json.load(f)
        print("Configuration chargée avec succès.")

        # --- Interprétation de k_range ---
        # La clé dans JSON est "clustering_k_range", on crée "clustering_k_range_obj" dans le dict
        if "clustering_k_range" in config_data and isinstance(config_data["clustering_k_range"], list) and len(config_data["clustering_k_range"]) == 2:
             k_start, k_end = config_data["clustering_k_range"]
             # Crée l'objet range python à partir de la liste [start, end] du JSON
             config_data["clustering_k_range_obj"] = range(k_start, k_end)
             print(f"  Interprétation de clustering_k_range: {list(config_data['clustering_k_range_obj'])}")
        else:
             print("Avertissement: 'clustering_k_range' non trouvé ou mal formaté (attendu [start, end]) dans config.json. Utilisation de range(2, 8) par défaut.")
             config_data["clustering_k_range_obj"] = range(2, 8) # Valeur par défaut si non trouvé/valide

        return config_data

    except FileNotFoundError: # Redondant car vérifié au début, mais gardé pour clarté
         print(f"Erreur : Fichier de configuration non trouvé (vérification interne) : {config_path}")
         raise
    except json.JSONDecodeError as e:
        print(f"Erreur FATALE lors du décodage du fichier JSON '{config_path}':")
        print(f"  Message: {e}")
        # Essayer d'afficher la ligne et la colonne de l'erreur si disponible
        print(f"  Erreur probable à la ligne {e.lineno}, colonne {e.colno}")
        # Afficher une partie du contenu autour de l'erreur peut aider
        try:
             with open(config_path, 'r', encoding='utf-8') as f_err:
                  lines = f_err.readlines()
                  start = max(0, e.lineno - 3)
                  end = min(len(lines), e.lineno + 2)
                  print("  Contenu autour de l'erreur :")
                  for i in range(start, end):
                       print(f"    Ligne {i+1}: {lines[i].rstrip()}")
        except Exception as read_err:
             print(f"  Impossible de lire le contenu du fichier pour le débogage: {read_err}")
        raise # Relancer l'exception pour arrêter
    except Exception as e:
        print(f"Erreur FATALE inattendue lors du chargement ou de l'interprétation de '{config_path}':")
        # Afficher la trace complète de l'erreur
        traceback.print_exc()
        raise # Relancer l'exception pour arrêter le pipeline


def run_pipeline(wsi_path: str, config: dict):
    """
    Exécute l'ensemble du pipeline d'analyse pour une seule WSI.

    Args:
        wsi_path (str): Chemin vers le fichier WSI d'entrée.
        config (dict): Dictionnaire contenant les paramètres de configuration chargés du JSON.
                       Doit inclure 'clustering_k_range_obj'.
    """
    start_time_total = time.time()
    print(f"--- Démarrage du Pipeline pour : {os.path.basename(wsi_path)} ---")

    # --- 1. Création du dossier de sortie spécifique à la lame ---
    wsi_filename = os.path.basename(wsi_path)
    wsi_name_no_ext = os.path.splitext(wsi_filename)[0]

    # Utiliser output_base_dir depuis la config chargée
    output_base_dir = config.get("output_base_dir", "pipeline_output_default") # Fallback au cas où
    slide_output_dir = os.path.join(output_base_dir, wsi_name_no_ext)

    # Définir les chemins spécifiques pour cette lame
    mask_output_path = os.path.join(slide_output_dir, "tissue_mask.png")
    patch_source_folder = os.path.join(slide_output_dir, "patches")
    feature_output_basename = os.path.join(slide_output_dir, "features", "extracted") # Pour .npy
    positions_file = f"{feature_output_basename}_positions.npy"
    features_file = f"{feature_output_basename}_features.npy"
    clustering_json_path = os.path.join(slide_output_dir, "clustering", "clustered_coordinates.json")
    clustered_patches_folder = os.path.join(slide_output_dir, "clustering", "clustered_patches")
    visualization_output_path = os.path.join(slide_output_dir, "clustering", "cluster_visualization.png")

    # Créer les dossiers nécessaires
    os.makedirs(slide_output_dir, exist_ok=True)
    os.makedirs(patch_source_folder, exist_ok=True)
    os.makedirs(os.path.dirname(feature_output_basename), exist_ok=True)
    os.makedirs(os.path.dirname(clustering_json_path), exist_ok=True)
    # Le dossier clustered_patches sera créé par organize_patches...

    print(f"Dossier de sortie : {slide_output_dir}")

    # --- 2. Génération du Masque Tissulaire ---
    print("\n--- Étape 1: Génération du Masque Tissulaire ---")
    start_time_step = time.time()
    determined_mask_level = -1 # Sera déterminé si level=-1
    tissue_mask = None
    try:
        mask_gen_level_config = config.get("mask_generation_level", -1)
        # Déterminer le niveau si nécessaire (pour generate_tissue_patches)
        if mask_gen_level_config == -1:
            # Utiliser un import dynamique pour openslide si possible, sinon standard
             try:
                 import openslide
             except ImportError:
                  # Tenter d'ajouter le chemin si défini (nécessaire sous Windows parfois)
                 if 'OPENSLIDE_PATH' in os.environ and os.path.exists(os.environ['OPENSLIDE_PATH']):
                     if hasattr(os, 'add_dll_directory'): # Windows Python 3.8+
                         with os.add_dll_directory(os.environ['OPENSLIDE_PATH']):
                              import openslide
                     else: # Solution de contournement pour autres OS/versions
                          os.environ['PATH'] = os.environ['OPENSLIDE_PATH'] + os.pathsep + os.environ['PATH']
                          import openslide
                 else:
                      raise ImportError("OpenSlide n'a pas pu être importé. Vérifiez l'installation ou définissez OPENSLIDE_PATH.")


             temp_slide = openslide.OpenSlide(wsi_path)
             target_max_dim = 2048 # Correspond à la logique dans generate_tissue_mask
             best_level = temp_slide.level_count - 1
             for i in range(temp_slide.level_count):
                 dims = temp_slide.level_dimensions[i]
                 if max(dims) <= target_max_dim:
                     best_level = i
                     break
             determined_mask_level = best_level
             temp_slide.close()
             print(f"Niveau de masque automatiquement déterminé : {determined_mask_level}")
        else:
             determined_mask_level = mask_gen_level_config

        tissue_mask = generate_tissue_mask(
            wsi_path=wsi_path,
            level=mask_gen_level_config, # Utilise -1 ou la valeur spécifiée
            threshold_method=config.get("mask_threshold_method", 'otsu'),
            kernel_size=config.get("mask_kernel_size", 7)
        )
        # Sauvegarder le masque
        cv2.imwrite(mask_output_path, tissue_mask)
        print(f"Masque sauvegardé dans : {mask_output_path}")
        print(f"Temps Étape 1 : {time.time() - start_time_step:.2f}s")

        # Vérifier si le masque est vide
        if np.sum(tissue_mask > 0) == 0:
             print("Avertissement : Le masque généré est entièrement noir. Aucun patch ne sera extrait.")
             # exit() # Décommenter pour arrêter si aucun tissu n'est trouvé

    except Exception as e:
        print(f"Erreur lors de la génération du masque : {e}")
        traceback.print_exc() # Afficher la trace pour le débogage
        return # Arrêter le pipeline pour cette lame

    # --- 3. Extraction des Patches ---
    patch_count = 0 # Initialiser avant le bloc conditionnel
    if tissue_mask is not None and np.sum(tissue_mask > 0) > 0: # Continuer seulement si masque valide
        print("\n--- Étape 2: Extraction des Patches ---")
        start_time_step = time.time()
        try:
            # Récupérer les paramètres depuis la config, avec des valeurs par défaut raisonnables
            patch_size = config.get("patch_size", 224)
            patch_stride = config.get("patch_stride", patch_size) # Stride par défaut = taille
            patch_extraction_level = config.get("patch_extraction_level", 0)
            # Utiliser get avec None comme défaut pour patches_max_to_process
            patches_max = config.get("patches_max_to_process", None) # null dans JSON -> None


            patch_generator = generate_tissue_patches(
                wsi_path=wsi_path,
                mask_array=tissue_mask,
                mask_level=determined_mask_level, # Utiliser le niveau déterminé
                patch_size=patch_size,
                stride=patch_stride,
                min_mask_coverage=config.get("patch_min_mask_coverage", 0.70),
                patch_extraction_level=patch_extraction_level
            )

            # Itérer et sauvegarder les patches générés
            for patch_np, x, y in patch_generator:
                # Vérifier la limite si elle est définie (différente de None)
                if patches_max is not None and patch_count >= patches_max:
                    print(f"Limite de {patches_max} patches atteinte.")
                    break

                # Nommer et sauvegarder le patch
                patch_filename = f"patch_L{patch_extraction_level}_x{x}_y{y}_s{patch_size}.png"
                patch_filepath = os.path.join(patch_source_folder, patch_filename)
                patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(patch_filepath, patch_bgr)
                patch_count += 1

            print(f"{patch_count} patches extraits et sauvegardés dans '{patch_source_folder}'.")
            print(f"Temps Étape 2 : {time.time() - start_time_step:.2f}s")

            if patch_count == 0:
                 print("Avertissement : Aucun patch n'a été extrait (vérifiez min_mask_coverage et le masque).")
                 # return # On pourrait arrêter ici

        except Exception as e:
            print(f"Erreur lors de l'extraction des patches : {e}")
            traceback.print_exc()
            return
    # else: # Pas besoin de else, patch_count reste 0


    # --- 4. Extraction des Caractéristiques ---
    positions_file_exists = False # Suivi si les fichiers .npy sont créés
    features_file_exists = False
    if patch_count > 0: # Continuer seulement si des patches ont été générés
        print("\n--- Étape 3: Extraction des Caractéristiques ---")
        start_time_step = time.time()
        try:
            # Configuration du device
            if not config.get("feature_force_cpu", False) and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            print(f"Utilisation du device : {device}")

            # Initialisation modèle et transform
            model, transform = setup_model_and_transform(config.get("feature_model_name", "hf-hub:bioptimus/H-optimus-0"), device)

            # Extraction
            extract_features(
                input_folder=patch_source_folder,
                output_basename=feature_output_basename, # Chemin de base pour .npy
                model=model,
                transform=transform,
                device=device,
                batch_size=config.get("feature_batch_size", 16)
            )
            print(f"Temps Étape 3 : {time.time() - start_time_step:.2f}s")

            # Vérifier si les fichiers .npy ont été créés
            positions_file_exists = os.path.exists(positions_file)
            features_file_exists = os.path.exists(features_file)
            if not positions_file_exists or not features_file_exists:
                 print("Erreur : Les fichiers de caractéristiques/positions n'ont pas été créés ou trouvés.")
                 # Ne pas retourner forcément, peut-être qu'on veut quand même le masque/patches

        except Exception as e:
            print(f"Erreur lors de l'extraction des caractéristiques : {e}")
            traceback.print_exc()
            # Ne pas retourner forcément

    # --- 5. Clustering ---
    final_labels = None # Initialiser pour les étapes suivantes
    positions = None # Initialiser
    clustering_success = False
    if positions_file_exists and features_file_exists: # Continuer seulement si les features existent
        print("\n--- Étape 4: Clustering ---")
        start_time_step = time.time()
        try:
            # Charger les données
            positions = np.load(positions_file)
            features = np.load(features_file).astype(np.float32) # Assurer float32

            if features.shape[0] == 0:
                 print("Aucune caractéristique à clusteriser.")
            else:
                 # Initialiser et exécuter le clustering
                 clusterer = Clustering()
                 # Utiliser l'objet range créé lors du chargement de la config JSON
                 k_range = config.get("clustering_k_range_obj", range(2, 8)) # Fallback
                 clustering_method = config.get("clustering_method", "BanditPam")

                 evaluation_scores = clusterer.evaluate_clusters(
                     features,
                     k_range=k_range,
                     method=clustering_method
                 )
                 if not evaluation_scores or np.all(np.isinf(evaluation_scores)):
                      print("Erreur : Aucun score d'évaluation valide. Impossible de déterminer k optimal.")
                 else:
                      optimal_k = clusterer.find_optimal_k(
                          evaluation_scores,
                          k_range=k_range, # Passer l'objet range
                          method=clustering_method
                      )
                      final_labels, _ = clusterer.apply_clustering(
                          features,
                          k=optimal_k,
                          method=clustering_method
                      )
                      # Sauvegarder les résultats JSON
                      clusterer.extract_labels(final_labels, positions, file_name=clustering_json_path)
                      clustering_success = True # Marquer le succès
                      print(f"Temps Étape 4 : {time.time() - start_time_step:.2f}s")

        except Exception as e:
            print(f"Erreur lors du clustering : {e}")
            traceback.print_exc()
            # final_labels reste None

    # --- 6. Organisation des Patches par Cluster ---
    # Utiliser clustering_success pour conditionner cette étape
    if clustering_success and final_labels is not None and positions is not None:
        print("\n--- Étape 5: Organisation des Patches par Cluster ---")
        start_time_step = time.time()
        try:
            organize_patches_by_cluster(
                cluster_labels=final_labels,
                positions=positions, # Doit être chargé de l'étape précédente
                patch_source_folder=patch_source_folder,
                clustered_output_folder=clustered_patches_folder,
                patch_size=config.get("patch_size", 224), # Utiliser la même taille que pour l'extraction
                patch_level=config.get("patch_extraction_level", 0) # Utiliser le même niveau
            )
            print(f"Temps Étape 5 : {time.time() - start_time_step:.2f}s")
        except Exception as e:
            print(f"Erreur lors de l'organisation des patches : {e}")
            traceback.print_exc()

    # --- 7. Visualisation des Clusters ---
    # Utiliser clustering_success pour conditionner cette étape
    if clustering_success and os.path.exists(clustering_json_path): # Continuer seulement si le JSON existe
        print("\n--- Étape 6: Visualisation des Clusters ---")
        start_time_step = time.time()
        try:
            visualize_clusters(
                json_path=clustering_json_path,
                output_image_path=visualization_output_path,
                title=f"{config.get('visualization_title', 'Visualisation')} ({wsi_name_no_ext})"
            )
            print(f"Temps Étape 6 : {time.time() - start_time_step:.2f}s")
        except Exception as e:
            print(f"Erreur lors de la visualisation des clusters : {e}")
            traceback.print_exc()


    end_time_total = time.time()
    print(f"\n--- Pipeline terminé pour : {wsi_filename} ---")
    print(f"Temps total d'exécution : {(end_time_total - start_time_total) / 60:.2f} minutes")
    print(f"Résultats disponibles dans : {slide_output_dir}")
    print("-" * 50)


if __name__ == "__main__":
    # Utiliser argparse pour l'argument WSI et optionnellement le fichier config JSON
    parser = argparse.ArgumentParser(description="Pipeline complet d'analyse d'images WSI.")
    parser.add_argument("wsi_path", help="Chemin vers le fichier image WSI (.svs, .tif, etc.).")
    parser.add_argument("-c", "--config", default="config.json", # Défaut = config.json
                        help="Chemin vers le fichier de configuration JSON. Défaut: config.json")

    args = parser.parse_args()

    if not os.path.exists(args.wsi_path):
        print(f"Erreur : Le fichier WSI spécifié n'existe pas : {args.wsi_path}")
        exit(1)

    # Charger la configuration depuis le fichier JSON spécifié (ou par défaut)
    try:
        PIPELINE_CONFIG = load_config_from_json(args.config)
    except Exception as e:
        # Erreur déjà affichée dans load_config_from_json ou message générique si autre erreur
        print(f"Erreur critique lors du chargement de la configuration. Arrêt.")
        exit(1)

    # Exécuter le pipeline principal avec la configuration chargée
    run_pipeline(args.wsi_path, PIPELINE_CONFIG)