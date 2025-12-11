OPENSLIDE_PATH = r"C:\Users\PATH/TO/OPENSLIDE\code\bin"

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import numpy as np
import math
from typing import Tuple, Generator, Union, Optional

def generate_tissue_patches(
    wsi_path: str,
    mask_array: np.ndarray,
    mask_level: int,
    patch_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    min_mask_coverage: float = 0.5,
    patch_extraction_level: int = 0
) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    """
    Génère des patches (régions) de l'image WSI originale qui se trouvent
    principalement dans les zones blanches (tissu) d'un masque binaire fourni.

    Args:
        wsi_path (str): Chemin vers le fichier WSI (par ex. .svs, .tif).
        mask_array (np.ndarray): Tableau NumPy 2D représentant le masque binaire.
                                 Les valeurs > 0 (typiquement 255 ou 1) indiquent le tissu.
        mask_level (int): Le niveau de résolution de la pyramide WSI auquel
                          `mask_array` a été généré.
        patch_size (Union[int, Tuple[int, int]]): Taille souhaitée des patches
                                                  (largeur, hauteur) à extraire au
                                                  `patch_extraction_level`. Si int, taille carrée.
        stride (Optional[Union[int, Tuple[int, int]]]): Pas de déplacement (stride) pour
                                                        générer les patches. Si int, stride
                                                        carré. Si None, le stride est égal
                                                        à `patch_size` (patches non chevauchants).
                                                        Doit être au `patch_extraction_level`.
        min_mask_coverage (float): Fraction minimale (entre 0 et 1) de la zone du patch
                                   (projetée sur le masque) qui doit être couverte par
                                   du tissu (valeur > 0 dans le masque) pour que le patch
                                   soit généré. Défaut à 0.5 (50%).
        patch_extraction_level (int): Niveau de la pyramide WSI auquel extraire les
                                      patches. Défaut à 0 (plus haute résolution).

    Yields:
        Generator[Tuple[np.ndarray, int, int], None, None]: Un générateur qui produit
            des tuples contenant:
            - patch_image (np.ndarray): Le patch extrait sous forme de tableau NumPy RGB.
            - x_coord (int): Coordonnée X du coin supérieur gauche du patch au niveau 0.
            - y_coord (int): Coordonnée Y du coin supérieur gauche du patch au niveau 0.

    Raises:
        FileNotFoundError: Si le fichier WSI n'est pas trouvé.
        openslide.OpenSlideError: Si le fichier ne peut pas être ouvert par OpenSlide.
        ValueError: Si les niveaux spécifiés sont invalides, si les dimensions du masque
                    ne correspondent pas, ou si `min_mask_coverage` est hors limites.
    """
    if not (0 <= min_mask_coverage <= 1):
        raise ValueError("min_mask_coverage doit être entre 0 et 1.")

    slide = None  # Initialiser pour le bloc finally
    try:
        slide = openslide.OpenSlide(wsi_path)

        # --- Validation des niveaux et dimensions ---
        if mask_level < 0 or mask_level >= slide.level_count:
            raise ValueError(f"Niveau de masque invalide: {mask_level}. Niveaux disponibles: 0-{slide.level_count - 1}")
        if patch_extraction_level < 0 or patch_extraction_level >= slide.level_count:
             raise ValueError(f"Niveau d'extraction de patch invalide: {patch_extraction_level}. Niveaux disponibles: 0-{slide.level_count - 1}")

        mask_dims_expected = slide.level_dimensions[mask_level]
        if mask_array.shape[0] != mask_dims_expected[1] or mask_array.shape[1] != mask_dims_expected[0]:
            raise ValueError(
                f"Les dimensions du masque fourni ({mask_array.shape[1]}, {mask_array.shape[0]}) "
                f"ne correspondent pas aux dimensions attendues pour le niveau {mask_level} "
                f"({mask_dims_expected[0]}, {mask_dims_expected[1]})"
            )

        # --- Normaliser patch_size et stride ---
        if isinstance(patch_size, int):
            patch_w, patch_h = patch_size, patch_size
        else:
            patch_w, patch_h = patch_size

        if stride is None:
            stride_x, stride_y = patch_w, patch_h
        elif isinstance(stride, int):
            stride_x, stride_y = stride, stride
        else:
            stride_x, stride_y = stride
            
        if stride_x <= 0 or stride_y <= 0:
             raise ValueError("Stride must be positive.")


        # --- Obtenir les informations de dimensions et de downsampling ---
        level0_w, level0_h = slide.level_dimensions[0]
        extraction_level_w, extraction_level_h = slide.level_dimensions[patch_extraction_level]
        
        # Facteur de réduction total entre le niveau d'extraction et le niveau du masque
        downsample_ex_to_mask = slide.level_downsamples[mask_level] / slide.level_downsamples[patch_extraction_level]
        
        # Facteur de réduction entre le niveau 0 et le niveau d'extraction
        downsample_0_to_ex = slide.level_downsamples[patch_extraction_level]


        print(f"Extraction des patches de {patch_w}x{patch_h} au niveau {patch_extraction_level} (dims: {extraction_level_w}x{extraction_level_h}).")
        print(f"Vérification par rapport au masque niveau {mask_level} (dims: {mask_dims_expected[0]}x{mask_dims_expected[1]}).")
        print(f"Stride: ({stride_x}, {stride_y}). Couverture minimale du masque: {min_mask_coverage:.1%}")


        # --- Itérer sur la grille de patches potentiels au NIVEAU D'EXTRACTION ---
        for y_ex in range(0, extraction_level_h - patch_h + 1, stride_y):
            for x_ex in range(0, extraction_level_w - patch_w + 1, stride_x):

                # --- Mapper les coordonnées du patch (niveau extraction) au niveau du masque ---
                # Coordonnées du centre du patch au niveau d'extraction
                center_x_ex = x_ex + patch_w / 2
                center_y_ex = y_ex + patch_h / 2

                # Coordonnées correspondantes au niveau du masque
                mask_x_start = math.floor(x_ex / downsample_ex_to_mask)
                mask_y_start = math.floor(y_ex / downsample_ex_to_mask)
                mask_x_end = math.ceil((x_ex + patch_w) / downsample_ex_to_mask)
                mask_y_end = math.ceil((y_ex + patch_h) / downsample_ex_to_mask)

                # Assurer que les coordonnées restent dans les limites du masque
                mask_x_start = max(0, mask_x_start)
                mask_y_start = max(0, mask_y_start)
                mask_x_end = min(mask_dims_expected[0], mask_x_end) # mask width
                mask_y_end = min(mask_dims_expected[1], mask_y_end) # mask height

                # Vérifier si la région résultante dans le masque a une taille valide
                if mask_x_end <= mask_x_start or mask_y_end <= mask_y_start:
                    continue # La projection du patch est en dehors ou trop petite sur le masque

                # --- Vérifier la couverture du masque ---
                # Extraire la petite région correspondante du masque
                mask_region = mask_array[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

                # Calculer la couverture (fraction de pixels > 0)
                coverage = np.mean(mask_region > 0) # Fonctionne si tissu = 1, 255, ou autre > 0

                if coverage >= min_mask_coverage:
                    # --- Le patch est valide, l'extraire ---
                    
                    # Coordonnées au niveau 0 (pour référence et lecture)
                    x0 = round(x_ex * downsample_0_to_ex)
                    y0 = round(y_ex * downsample_0_to_ex)

                    # Lire la région depuis le WSI au niveau d'extraction spécifié
                    patch_pil = slide.read_region(
                        location=(x0, y0), # La localisation est toujours en coordonnées niveau 0
                        level=patch_extraction_level,
                        size=(patch_w, patch_h)
                    )

                    # Convertir en NumPy array RGB
                    patch_np = np.array(patch_pil.convert('RGB'))
                    
                    # Coordonnées retournées sont celles au niveau 0 pour une référence absolue
                    yield patch_np, x0, y0

    except FileNotFoundError:
        print(f"Erreur: Fichier WSI non trouvé - {wsi_path}")
        raise
    except openslide.OpenSlideError as e:
        print(f"Erreur: Impossible d'ouvrir/lire le fichier WSI {wsi_path} - {e}")
        raise
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")
        raise
    finally:
        # Toujours fermer le slide
        if slide:
            slide.close()
            # print("Slide fermé.")


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    import cv2 # Pour sauvegarder/générer un masque d'exemple
    import os
    from mask_extractor import generate_tissue_mask # Supposons que la fonction précédente est dans tissue_masker.py

    # --- Paramètres ---
    # ** REMPLACEZ CECI par le chemin vers votre fichier WSI **
    # exemple_wsi = "chemin/vers/votre/image.svs"
    exemple_wsi = "../images/image_test.svs" # Mettez un vrai chemin ici !

    output_patch_dir = "output_patches"
    os.makedirs(output_patch_dir, exist_ok=True)

    patch_size_px = 224  # Taille des patches à extraire (au niveau 0)
    patches_max_to_save = 200000 # Limiter le nombre de patches sauvegardés pour cet exemple

    # --- Étape 1: Générer ou Charger le Masque ---
    # (Assurez-vous que tissue_masker.py est dans le même dossier ou accessible)
    mask_generation_level = -1 # Laisser la fonction choisir un niveau bas (-1)
    mask_file_path = "output/tissue_mask_otsu.png" # Chemin où le masque a été sauvegardé précédemment

    try:
        # Essayer de charger le masque s'il existe déjà
        if os.path.exists(mask_file_path):
            print(f"Chargement du masque existant: {mask_file_path}")
            # Lire en niveaux de gris, car c'est un masque binaire
            tissue_mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
            
            # Récupérer le niveau auquel il a été généré (nécessite de le savoir/sauvegarder)
            # Pour cet exemple, on suppose qu'il a été généré au niveau choisi par generate_tissue_mask
            # Nous devons ouvrir temporairement le slide pour connaître ce niveau
            temp_slide = openslide.OpenSlide(exemple_wsi)
            target_max_dim = 2048
            determined_mask_level = temp_slide.level_count - 1
            for i in range(temp_slide.level_count):
                dims = temp_slide.level_dimensions[i]
                if max(dims) <= target_max_dim:
                    determined_mask_level = i
                    break
            temp_slide.close()
            print(f"Niveau du masque chargé déterminé comme étant: {determined_mask_level}")
            mask_level_used = determined_mask_level

        else:
             # Si le masque n'existe pas, le générer
             print("Génération d'un nouveau masque (méthode Otsu)...")
             # Note : Il faut savoir à quel niveau le masque est généré !
             # Ouvrir temporairement pour déterminer le niveau choisi par level=-1
             temp_slide = openslide.OpenSlide(exemple_wsi)
             target_max_dim = 2048
             determined_mask_level = temp_slide.level_count - 1
             for i in range(temp_slide.level_count):
                 dims = temp_slide.level_dimensions[i]
                 if max(dims) <= target_max_dim:
                     determined_mask_level = i
                     break
             temp_slide.close()
             mask_level_used = determined_mask_level
             
             tissue_mask = generate_tissue_mask(exemple_wsi, level=mask_level_used, threshold_method='otsu')
             # Optionnel: sauvegarder le masque généré
             os.makedirs(os.path.dirname(mask_file_path), exist_ok=True)
             cv2.imwrite(mask_file_path, tissue_mask)
             print(f"Nouveau masque généré au niveau {mask_level_used} et sauvegardé dans {mask_file_path}")

        if tissue_mask is None:
             raise Exception(f"Impossible de charger ou générer le masque binaire.")
        if np.sum(tissue_mask > 0) == 0:
             print("Avertissement : Le masque chargé/généré est entièrement noir (pas de tissu détecté). Aucun patch ne sera extrait.")
             exit()


        # --- Étape 2: Générer et Sauvegarder les Patches ---
        print("\nDémarrage de la génération des patches...")
        patch_count = 0
        # Utiliser le générateur pour traiter les patches un par un
        # Extraire les patches au niveau 0, stride = taille du patch (non-chevauchant)
        # Exiger qu'au moins 70% du patch (sur le masque) soit du tissu
        patch_generator = generate_tissue_patches(
            wsi_path=exemple_wsi,
            mask_array=tissue_mask,
            mask_level=mask_level_used,
            patch_size=patch_size_px,
            stride=patch_size_px, # Non-chevauchant
            min_mask_coverage=0.1, # Au moins 10% dans le masque
            patch_extraction_level=0 # Extraire à la plus haute résolution
        )

        for patch_np, x, y in patch_generator:
            if patch_count >= patches_max_to_save:
                print(f"Limite de {patches_max_to_save} patches sauvegardés atteinte.")
                break

            # Nommer le fichier patch avec ses coordonnées niveau 0
            patch_filename = f"patch_L{0}_x{x}_y{y}_s{patch_size_px}.png"
            patch_filepath = os.path.join(output_patch_dir, patch_filename)

            # Sauvegarder le patch (OpenCV attend BGR, mais PIL/Numpy donne RGB, donc convertir)
            patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(patch_filepath, patch_bgr)

            patch_count += 1
            if patch_count % 10 == 0: # Afficher la progression
                print(f"  {patch_count} patches sauvegardés...")

        print(f"\nTerminé. {patch_count} patches sauvegardés dans '{output_patch_dir}'.")

    except FileNotFoundError:
        print(f"\nErreur: Le fichier WSI d'exemple '{exemple_wsi}' n'a pas été trouvé.")
        print("Veuillez remplacer 'placeholder_image.svs' par le chemin réel de votre fichier WSI.")
    except openslide.OpenSlideError as e:
         print(f"\nErreur OpenSlide : {e}")
         print("Assurez-vous que le fichier est un WSI valide et que OpenSlide (librairie C et Python) est correctement installé.")
    except ValueError as e:
         print(f"\nErreur de valeur/configuration : {e}")
    except ImportError:
         print("\nErreur: La fonction 'generate_tissue_mask' n'a pas pu être importée.")
         print("Assurez-vous que le fichier 'tissue_masker.py' est accessible.")
    except Exception as e:
        print(f"\nUne erreur inattendue est survenue dans l'exemple : {type(e).__name__} - {e}")