# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Pour gérer les couleurs
import os

def visualize_clusters(json_path: str, output_image_path: str, title: str = "Visualisation des Clusters de Patches"):
    """
    Crée un scatter plot pour visualiser les clusters de patches.

    Args:
        json_path (str): Chemin vers le fichier JSON contenant les coordonnées par cluster
                         (généré par le script de clustering).
        output_image_path (str): Chemin où sauvegarder l'image du graphique généré (ex: 'cluster_visualization.png').
        title (str): Titre du graphique.
    """
    # --- 1. Charger les données du JSON ---
    print(f"Chargement des données de clustering depuis : {json_path}")
    try:
        with open(json_path, 'r') as f:
            # Charger les clés comme entiers si elles ont été sauvegardées comme chaînes
            data = json.load(f)
            # Convertir les clés en entiers et les coordonnées en array numpy si nécessaire
            cluster_coords = {}
            for k, v in data.items():
                 try:
                     cluster_id = int(k)
                     # Convertir chaque coordonnée [x,y] en un array numpy
                     cluster_coords[cluster_id] = np.array(v, dtype=int)
                 except ValueError:
                      print(f"Avertissement : Impossible de convertir la clé de cluster '{k}' en entier. Ignoré.")
                      continue
                 except Exception as e_inner:
                      print(f"Avertissement : Erreur lors du traitement des coordonnées pour le cluster '{k}': {e_inner}. Ignoré.")
                      continue

        if not cluster_coords:
             print("Erreur : Aucune donnée de cluster valide trouvée dans le fichier JSON.")
             return

        print(f"{len(cluster_coords)} clusters chargés.")

    except FileNotFoundError:
        print(f"Erreur : Fichier JSON non trouvé à '{json_path}'.")
        return
    except json.JSONDecodeError:
        print(f"Erreur : Impossible de décoder le fichier JSON à '{json_path}'. Vérifiez son format.")
        return
    except Exception as e:
        print(f"Erreur inattendue lors du chargement du JSON : {e}")
        return

    # --- 2. Préparer les données pour le plotting ---
    all_x = []
    all_y = []
    all_labels = []
    cluster_ids = sorted(cluster_coords.keys()) # Trier pour une légende cohérente

    print("Préparation des données pour le graphique...")
    for cluster_id in cluster_ids:
        coords = cluster_coords[cluster_id]
        if coords.ndim == 2 and coords.shape[1] == 2: # S'assurer que ce sont des paires [x, y]
            all_x.extend(coords[:, 0]) # Coordonnées X
            all_y.extend(coords[:, 1]) # Coordonnées Y
            all_labels.extend([cluster_id] * len(coords)) # Répéter l'ID du cluster
        else:
            print(f"Avertissement: Données de coordonnées invalides pour le cluster {cluster_id}. Attendu un array Nx2.")


    if not all_x:
        print("Erreur : Aucune coordonnée valide à afficher.")
        return

    num_clusters = len(cluster_ids)
    print(f"Total de {len(all_x)} points à afficher pour {num_clusters} clusters.")

    # --- 3. Créer le Scatter Plot ---
    print("Création du graphique...")
    plt.figure(figsize=(12, 10)) # Ajuster la taille pour une meilleure lisibilité

    # Choisir une palette de couleurs distinctes
    if num_clusters <= 10:
        colors = cm.get_cmap('tab10', num_clusters)
    elif num_clusters <= 20:
        colors = cm.get_cmap('tab20', num_clusters)
    else:
        colors = cm.get_cmap('viridis', num_clusters)
        print("Avertissement: Plus de 20 clusters, les couleurs peuvent être difficiles à distinguer.")

    # Créer le scatter plot, en colorant par label de cluster
    scatter = plt.scatter(all_x, all_y, c=all_labels, cmap=colors, s=10, alpha=0.7) # s=taille, alpha=transparence

    # --- 4. Ajouter Légende, Titre et Labels ---
    plt.title(title, fontsize=16)
    plt.xlabel("Coordonnée X du Patch (Niveau 0)", fontsize=12)
    plt.ylabel("Coordonnée Y du Patch (Niveau 0)", fontsize=12)

    # Créer une légende personnalisée
    handles, _ = scatter.legend_elements(prop="colors", num=num_clusters)
    legend_labels = [f'Cluster {cid}' for cid in cluster_ids]

    plt.legend(handles, legend_labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().invert_yaxis() # Inverser l'axe Y

    # Ajuster la mise en page pour éviter que la légende ne sorte
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # --- 5. Sauvegarder le Graphique ---
    try:
        # Créer le dossier parent si nécessaire
        output_dir = os.path.dirname(output_image_path)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir)

        print(f"Sauvegarde du graphique dans : {output_image_path}")
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print("Graphique sauvegardé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du graphique : {e}")

    # Optionnel: fermer la figure pour libérer la mémoire si on génère beaucoup de graphiques
    plt.close()


if __name__ == "__main__":
    # --- Configuration (chemins définis directement) ---
    JSON_INPUT_PATH = "clustered_coordinates.json"
    OUTPUT_IMAGE_PATH = "cluster_visualization.png"
    GRAPH_TITLE = "Visualisation des Clusters de Patches"

    # --- Exécution de la visualisation ---
    visualize_clusters(
        json_path=JSON_INPUT_PATH,
        output_image_path=OUTPUT_IMAGE_PATH,
        title=GRAPH_TITLE
    )

    print("\nScript de visualisation terminé.")