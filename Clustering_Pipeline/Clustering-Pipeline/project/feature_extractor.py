# -*- coding: utf-8 -*-
import os
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import login, HfFolder
from tqdm import tqdm # Pour la barre de progression
# argparse n'est plus nécessaire
import re # Pour l'extraction des coordonnées des noms de fichiers

def setup_model_and_transform(model_name="hf-hub:bioptimus/H-optimus-0", device="cuda"):
    """
    Initialise le modèle et les transformations d'image.

    Args:
        model_name (str): Nom du modèle à charger depuis timm/Hugging Face Hub.
        device (str): Périphérique sur lequel charger le modèle ('cuda' ou 'cpu').

    Returns:
        tuple: (modèle, transformation)
    """
    print(f"Chargement du modèle : {model_name}")
    model = timm.create_model(
        model_name,
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False
    )
    model.to(device)
    model.eval()
    print("Modèle chargé et mis en mode évaluation.")

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517)
        ),
    ])
    print("Transformations d'image définies.")
    return model, transform

def parse_coordinates_from_filename(filename: str) -> tuple[int, int] | None:
    """
    Extrait les coordonnées X et Y d'un nom de fichier de patch.
    Attend un format comme 'patch_L{level}_x{X}_y{Y}_s{size}.png'.

    Args:
        filename (str): Le nom de base du fichier image.

    Returns:
        tuple[int, int] | None: Un tuple (x, y) si l'extraction réussit, sinon None.
    """
    # Utiliser une expression régulière pour extraire x et y
    match = re.search(r'_x(\d+)_y(\d+)_', filename)
    if match:
        try:
            x = int(match.group(1))
            y = int(match.group(2))
            return x, y
        except ValueError:
            # Au cas où les groupes capturés ne sont pas des entiers valides
            # print(f"Avertissement : Impossible de convertir les coordonnées en entiers pour {filename}")
            return None
    else:
        # print(f"Avertissement : Format de nom de fichier non reconnu pour l'extraction des coordonnées : {filename}")
        return None

def extract_features(input_folder: str, output_basename: str, model, transform, device: str, batch_size: int = 8):
    """
    Extrait les caractéristiques des images d'un dossier et les sauvegarde
    dans deux fichiers .npy (positions et caractéristiques).

    Args:
        input_folder (str): Chemin vers le dossier contenant les images.
        output_basename (str): Nom de base pour les fichiers de sortie .npy.
                               (ex: 'features' -> 'features_positions.npy', 'features_features.npy').
        model: Le modèle timm pré-chargé.
        transform: Les transformations torchvision à appliquer.
        device (str): Périphérique ('cuda' ou 'cpu').
        batch_size (int): Nombre d'images à traiter par lot.
    """
    if not os.path.isdir(input_folder):
        print(f"Erreur : Le dossier d'entrée '{input_folder}' n'existe pas.")
        return

    image_files = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    print(f"Recherche des images dans : {input_folder}")
    # Trier les fichiers pour un ordre potentiellement plus cohérent (optionnel)
    try:
        filenames = sorted(os.listdir(input_folder))
    except Exception as e:
        print(f"Erreur lors de la lecture du dossier {input_folder}: {e}")
        return

    for fname in filenames:
        if fname.lower().endswith(valid_extensions):
            image_files.append(os.path.join(input_folder, fname))

    if not image_files:
        print("Aucune image trouvée dans le dossier.")
        return

    print(f"{len(image_files)} images trouvées. Début de l'extraction des caractéristiques...")

    all_positions_list = []
    all_features_list = []
    skipped_files = 0

    model.eval() # Assurer que le modèle est en mode évaluation

    # Traitement par lots pour l'efficacité
    for i in tqdm(range(0, len(image_files), batch_size), desc="Traitement des lots"):
        batch_files = image_files[i:i+batch_size]
        batch_tensors = []
        valid_files_in_batch = [] # Garder trace des fichiers valides dans le lot
        coords_in_batch = []      # Coordonnées correspondantes

        # Préparer le lot et vérifier les noms de fichiers
        for img_path in batch_files:
            filename = os.path.basename(img_path)
            coords = parse_coordinates_from_filename(filename)
            if coords is None:
                # print(f"Skipping file due to coordinate parsing issue: {filename}")
                skipped_files += 1
                continue # Ignorer ce fichier

            try:
                # Utiliser Image.LANCZOS pour une meilleure qualité de redimensionnement si nécessaire
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img)
                batch_tensors.append(tensor)
                valid_files_in_batch.append(img_path)
                coords_in_batch.append(coords)
            except Exception as e:
                print(f"Erreur lors du chargement/traitement de l'image {filename} : {e}")
                skipped_files += 1
                # On ignore cette image et on continue avec les autres du lot

        if not batch_tensors: # Si aucune image valide dans le lot
             continue

        # Empiler les tenseurs pour former un lot
        batch_input = torch.stack(batch_tensors).to(device)

        # Inférence avec mixed precision si CUDA est disponible
        try:
            with torch.inference_mode(): # Désactive le calcul de gradient
                if device == "cuda":
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        features = model(batch_input)
                else:
                    features = model(batch_input)

            # Détacher les features du GPU, les mettre sur CPU
            # Convertir en float32 pour une meilleure compatibilité avec numpy
            features_cpu = features.detach().cpu().to(torch.float32)

            # Ajouter les résultats aux listes globales
            all_positions_list.extend(coords_in_batch)
            all_features_list.extend(list(features_cpu)) # Ajouter chaque tenseur de feature individuellement

        except Exception as e:
            print(f"Erreur lors de l'inférence sur le lot commençant par {os.path.basename(valid_files_in_batch[0])}: {e}")
            # Vous pourriez vouloir ajouter une gestion d'erreur plus fine ici

    # --- Sauvegarder les caractéristiques et positions ---
    if skipped_files > 0:
        print(f"Avertissement : {skipped_files} fichier(s) ont été ignorés à cause d'erreurs de chargement ou de parsing du nom.")

    if all_features_list:
        num_features = len(all_features_list)
        print(f"\nExtraction terminée. {num_features} vecteurs de caractéristiques extraits.")

        # Convertir les listes en tableaux NumPy
        positions_array = np.array(all_positions_list, dtype=int)
        # Empiler les tenseurs de caractéristiques en un seul tableau NumPy
        features_array = torch.stack(all_features_list).numpy()

        # Définir les noms de fichiers de sortie
        positions_output_file = f"{output_basename}_positions.npy"
        features_output_file = f"{output_basename}_features.npy"

        # Créer le dossier de sortie s'il n'existe pas
        output_dir = os.path.dirname(positions_output_file)
        if output_dir and not os.path.exists(output_dir):
             try:
                 os.makedirs(output_dir)
                 print(f"Dossier de sortie créé : {output_dir}")
             except OSError as e:
                 print(f"Erreur lors de la création du dossier de sortie {output_dir}: {e}")
                 # Continuer quand même, la sauvegarde échouera probablement mais on aura un message d'erreur clair

        print(f"Sauvegarde des positions dans : {positions_output_file} (shape: {positions_array.shape})")
        print(f"Sauvegarde des caractéristiques dans : {features_output_file} (shape: {features_array.shape})")

        try:
            np.save(positions_output_file, positions_array)
            np.save(features_output_file, features_array)
            print("Sauvegarde réussie.")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des fichiers .npy : {e}")
    else:
        print("\nAucune caractéristique n'a été extraite (vérifiez les images et les noms de fichiers).")


if __name__ == "__main__":
    # --- Définition des paramètres directement dans le code ---
    INPUT_FOLDER = "output_patches"
    OUTPUT_BASENAME = "extracted" # Sera utilisé pour créer extracted_positions.npy et extracted_features.npy
    MODEL_NAME = "hf-hub:bioptimus/H-optimus-0"
    BATCH_SIZE = 16
    FORCE_CPU = False # Mettre à True pour forcer l'utilisation du CPU

    # --- Connexion à Hugging Face Hub ---
    print("Tentative de connexion à Hugging Face Hub...")
    try:
        if HfFolder.get_token() is None:
             print("Token Hugging Face non trouvé localement. Tentative de connexion interactive.")
             login()
        else:
             print("Utilisation du token Hugging Face trouvé localement.")
    except Exception as e:
        print(f"Avertissement : Échec de la connexion à Hugging Face Hub : {e}")
        print("Le script continuera, mais le téléchargement du modèle pourrait échouer s'il n'est pas en cache.")

    # --- Configuration du Périphérique ---
    if not FORCE_CPU and torch.cuda.is_available():
        DEVICE = "cuda"
        print("CUDA est disponible. Utilisation du GPU.")
    else:
        DEVICE = "cpu"
        if FORCE_CPU:
             print("Utilisation du CPU forcée.")
        else:
             print("CUDA non disponible. Utilisation du CPU.")

    # --- Initialisation ---
    try:
        model, transform = setup_model_and_transform(MODEL_NAME, DEVICE)
    except Exception as e:
        print(f"Erreur lors de l'initialisation du modèle ou des transformations : {e}")
        exit(1)

    # --- Extraction ---
    extract_features(
        input_folder=INPUT_FOLDER,
        output_basename=OUTPUT_BASENAME,
        model=model,
        transform=transform,
        device=DEVICE,
        batch_size=BATCH_SIZE
    )

    print("\nScript terminé.")